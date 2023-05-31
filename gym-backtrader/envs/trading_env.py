import logging
import random
from abc import abstractmethod
from enum import Enum
from typing import Optional, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from pandas import DataFrame



class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4



class Positions(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.

def transform(position: Positions, action: int):
    '''
    Overview:
        used by env.tep().
        This func is used to transform the env's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Doulbe_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    '''
    if action == Actions.SELL:

        if position == Positions.LONG:
            return Positions.FLAT, False

        if position == Positions.FLAT:
            return Positions.SHORT, True

    if action == Actions.BUY:

        if position == Positions.SHORT:
            return Positions.FLAT, False

        if position == Positions.FLAT:
            return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and (position == Positions.LONG or position == Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and (position == Positions.SHORT or position == Positions.FLAT):
        return Positions.LONG, True

    return position, False



class TradingEnv(gym.Env):

    def __init__(self, df: DataFrame = DataFrame(), prices: DataFrame = DataFrame(),
                 reward_kwargs: dict = {}, window_size=10, starting_point=True,
                 id: str = 'baseenv-1', seed: int = 1, config: dict = {},
                 fee: float = 0.0015,  pair: str = "",
                 df_raw: DataFrame = DataFrame()):
        """
        Initializes the training/eval environment.
        :param df: dataframe of features
        :param prices: dataframe of prices to be used in the training environment
        :param window_size: size of window (temporal) to pass to the agent
        :param reward_kwargs: extra config settings assigned by user in `rl_config`
        :param starting_point: start at edge of window or not
        :param id: string id of the environment (used in backend for multiprocessed env)
        :param seed: Sets the seed of the environment higher in the gym.Env object
        :param config: Typical user configuration file
        :param live: Whether or not this environment is active in dry/live/backtesting
        :param fee: The fee to use for environmental interactions.
        :param can_short: Whether or not the environment can short
        """
        self.config: dict = config
        self.rl_config: dict = config['freqai']['rl_config']
        self.add_state_info: bool = self.rl_config.get('add_state_info', False)
        self.id: str = id
        self.max_drawdown: float = 1 - self.rl_config.get('max_training_drawdown_pct', 0.8)
        self.compound_trades: bool = config['stake_amount'] == 'unlimited'
        self.pair: str = pair
        self.raw_features: DataFrame = df_raw
        if self.config.get('fee', None) is not None:
            self.fee = self.config['fee']
        else:
            self.fee = fee

        # set here to default 5Ac, but all children envs can override this
        self.actions: Type[Enum] = Actions
        self.tensorboard_metrics: dict = {}
        self.seed(seed)
        self.reset_env(df, prices, window_size, reward_kwargs, starting_point)

    def reset_env(self, df: DataFrame, prices: DataFrame, window_size: int,
                  reward_kwargs: dict, starting_point=True):
        """
        Resets the environment when the agent fails (in our case, if the drawdown
        exceeds the user set max_training_drawdown_pct)
        :param df: dataframe of features
        :param prices: dataframe of prices to be used in the training environment
        :param window_size: size of window (temporal) to pass to the agent
        :param reward_kwargs: extra config settings assigned by user in `rl_config`
        :param starting_point: start at edge of window or not
        """
        self.signal_features: DataFrame = df
        self.prices: DataFrame = prices
        self.window_size: int = window_size
        self.starting_point: bool = starting_point
        self.rr: float = reward_kwargs["rr"]
        self.profit_aim: float = reward_kwargs["profit_aim"]

        # # spaces
        if self.add_state_info:
            self.total_features = self.signal_features.shape[1] + 3
        else:
            self.total_features = self.signal_features.shape[1]
        self.shape = (window_size, self.total_features)
        self.set_action_space()
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.prices) - 1
        self._done: bool = False
        self._current_tick: int = self._start_tick
        self._last_trade_tick: Optional[int] = None
        self._position = Positions.FLAT
        self._position_history: list = [None]
        self.total_reward: float = 0
        self._total_profit: float = 1
        self._total_unrealized_profit: float = 1
        self.history: dict = {}



    def get_attr(self, attr: str):
        """
        Returns the attribute of the environment
        :param attr: attribute to return
        :return: attribute
        """
        return getattr(self, attr)

    def set_action_space(self):
        self.action_space = spaces.Discrete(len(Actions))

    def seed(self, seed: int = 1):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        """
        Reset is called at the beginning of every episode
        """

        self._done = False

        if self.starting_point is True:
            if self.rl_config.get('randomize_starting_position', False):
                length_of_data = int(self._end_tick / 4)
                start_tick = random.randint(self.window_size + 1, length_of_data)
                self._start_tick = start_tick
            self._position_history = (self._start_tick * [None]) + [self._position]
        else:
            self._position_history = (self.window_size * [None]) + [self._position]

        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.FLAT

        self.total_reward = 0.
        self._total_profit = 1.  # unit
        self.history = {}
        self.portfolio_log_returns = np.zeros(len(self.prices))

        self._profits = [(self._start_tick, 1)]
        self.close_trade_profit = []
        self._total_unrealized_profit = 1

        return self._get_observation(), self.history

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._update_unrealized_total_profit()
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward

        trade_type = None
        self._position, trade = transform(self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        if (self._total_profit < self.max_drawdown or
                self._total_unrealized_profit < self.max_drawdown):
            self._done = True

        self._position_history.append(self._position)

        info = dict(
            tick=self._current_tick,
            action=action,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            trade_duration=self.get_trade_duration(),
            current_profit_pct=self.get_unrealized_profit()
        )

        observation = self._get_observation()
        # user can play with time if they want
        truncated = False

        self._update_history(info)

        return observation, step_reward, self._done, truncated, info


    def _get_observation(self):
        features_window = self.signal_features[(
            self._current_tick - self.window_size):self._current_tick]
        if self.add_state_info:
            features_and_state = DataFrame(np.zeros((len(features_window), 3)),
                                           columns=['current_profit_pct',
                                                    'position',
                                                    'trade_duration'],
                                           index=features_window.index)

            features_and_state['current_profit_pct'] = self.get_unrealized_profit()
            features_and_state['position'] = self._position.value
            features_and_state['trade_duration'] = self.get_trade_duration()
            features_and_state = pd.concat([features_window, features_and_state], axis=1)
            return features_and_state
        else:
            return features_window


    def get_trade_duration(self):
        """
        Get the trade duration if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0
        else:
            return self._current_tick - self._last_trade_tick

    def get_unrealized_profit(self):
        """
        Get the unrealized profit if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.

        if self._position == Positions.FLAT:
            return 0.
        elif self._position == Positions.SHORT:
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.LONG:
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price) / last_trade_price
        else:
            return 0.

    def add_entry_fee(self, price):
        return price * (1 + self.fee)

    def add_exit_fee(self, price):
        return price / (1 + self.fee)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _update_unrealized_total_profit(self):
        """
        Update the unrealized total profit incase of episode end.
        """
        if self._position in (Positions.LONG, Positions.SHORT):
            pnl = self.get_unrealized_profit()
            if self.compound_trades:
                # assumes unit stake and compounding
                unrl_profit = self._total_profit * (1 + pnl)
            else:
                # assumes unit stake and no compounding
                unrl_profit = self._total_profit + pnl
            self._total_unrealized_profit = unrl_profit

    def _update_total_profit(self):
        pnl = self.get_unrealized_profit()
        if self.compound_trades:
            # assumes unit stake and compounding
            self._total_profit = self._total_profit * (1 + pnl)
        else:
            # assumes unit stake and no compounding
            self._total_profit += pnl

    def current_price(self) -> float:
        return self.prices.iloc[self._current_tick].open

    def get_actions(self) -> Type[Enum]:
        """
        Used by SubprocVecEnv to get actions from
        initialized env for tensorboard callback
        """
        return self.actions


    def calculate_reward(self, action: int) -> np.float32:
        step_reward = 0.
        current_price = (self.prices[self._current_tick])
        last_trade_price = (self.prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.fee) * (1 - self.fee))

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)

        return step_reward


