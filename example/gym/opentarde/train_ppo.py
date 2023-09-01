""""""
import numpy as np
import pandas as pd
import gymnasium as gym

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from pandas_ta.statistics import zscore
import gym_trading_env

windows_size = 30

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )


def reward_sortino_function(history):
    returns = pd.Series(history["portfolio_valuation"][-(windows_size+1):]).pct_change().dropna()
    downside_returns = returns.copy()
    downside_returns[returns < 0] = returns ** 2
    expected_return = returns.mean()
    downside_std = np.sqrt(np.std(downside_returns))
    if downside_std == 0 :
        return 0
    return (expected_return + 1E-9) / (downside_std + 1E-9)

def reward_sharpe_function(history):
    returns = np.diff(history["portfolio_valuation"])
    return (returns.mean() - 0) / (returns.std() + 1E-9)

def max_drawdown(history):
    networth_array = history['portfolio_valuation']
    _max_networth = networth_array[0]
    _max_drawdown = 0
    for networth in networth_array:
        if networth > _max_networth:
            _max_networth = networth
        drawdown = ( networth - _max_networth ) / _max_networth
        if drawdown < _max_drawdown:
            _max_drawdown = drawdown
    return f"{_max_drawdown*100:5.2f}%"


def create_env():
    df = pd.read_csv("../data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['feature_z_close'] = zscore(df['close'], length=windows_size)
    df['feature_z_open'] = zscore(df['open'], length=windows_size)
    df['feature_z_high'] = zscore(df['high'], length=windows_size)
    df['feature_z_low'] = zscore(df['low'], length=windows_size)
    df['feature_z_volume'] = zscore(df['volume'], length=windows_size)
    df.dropna(inplace=True)

    env = make(
            "TradingEnv",
             env_num=9,
            name= "BTCUSD",
            df = df,
            windows= windows_size,
            positions = [ -1, -0.5, 0, 0.5, 1], # From -1 (=SHORT), to +1 (=LONG)
            # initial_position = 'random', #Initial position
            initial_position=0,  # Initial position
            trading_fees = 0.1/100, # 0.01% per stock buy / sell
            borrow_interest_rate= 0, #per timestep (= 1h here)
            reward_function = reward_function,
            portfolio_initial_value = 10000, # in FIAT (here, USD)
            #max_episode_duration = 2400,
            #max_episode_duration=500,
        )
    # env.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    # env.add_metric('Max Drawdown', max_drawdown)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)
    return env


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment, set environment parallelism to 9
    # env = make("CartPole-v1", env_num=9)
    env = create_env()

    net = Net(
        env,
        cfg=cfg,
    )
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=20000)

    env.close()
    return agent


def evaluation(agent):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = "group_human"
    render_mode = None
    env = make("CartPole-v1", render_mode=render_mode, env_num=9, asynchronous=True)
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False
    step = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
    env.close()


if __name__ == "__main__":
    agent = train()
    # evaluation(agent)
