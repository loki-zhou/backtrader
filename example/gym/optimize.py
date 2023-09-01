import os
import pandas as pd

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import numpy as np
import gymnasium as gym
import custom_indicator as cta
from gym_trading_env.environments import TradingEnv
from pandas_ta.statistics import zscore

windows_size = 100

df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

# cta.NormalizedScore(df, windows_size)

df['feature_z_close'] = zscore(df['close'], length=windows_size )
df['feature_z_open'] = zscore(df['open'], length=windows_size )
df['feature_z_high'] = zscore(df['high'], length=windows_size )
df['feature_z_low'] = zscore(df['low'], length=windows_size )
df['feature_z_volume'] = zscore(df['volume'], length=windows_size )

df.dropna(inplace= True)

monitor_dir = ""


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
    env = gym.make(
            "TradingEnv",
            name= "BTCUSD",
            df = df,
            windows= 30,
            positions = [ -1, -0.5, 0, 0.5, 1], # From -1 (=SHORT), to +1 (=LONG)
            # initial_position = 'random', #Initial position
            initial_position=0,  # Initial position
            trading_fees = 0.1/100, # 0.01% per stock buy / sell
            borrow_interest_rate= 0, #per timestep (= 1h here)
            reward_function = reward_sortino_function,
            portfolio_initial_value = 10000, # in FIAT (here, USD)
            #max_episode_duration = 2400,
            #max_episode_duration=500,
        )
    env.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.add_metric('Max Drawdown', max_drawdown)
    env = Monitor(env, monitor_dir)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.NormalizeReward(env)
    return env


def create_rnn_env():
    env = gym.make(
            "TradingEnv",
            name= "BTCUSD",
            df = df,
            windows= 1,
            positions = [ -1, -0.5, 0, 0.5, 1], # From -1 (=SHORT), to +1 (=LONG)
            # initial_position = 'random', #Initial position
            initial_position=0,  # Initial position
            trading_fees = 0.1/100, # 0.01% per stock buy / sell
            borrow_interest_rate= 0, #per timestep (= 1h here)
            reward_function = reward_sortino_function,
            portfolio_initial_value = 10000, # in FIAT (here, USD)
            #max_episode_duration = 2400,
            #max_episode_duration=500,
        )
    env.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.add_metric('Max Drawdown', max_drawdown)
    env = Monitor(env, monitor_dir)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.NormalizeReward(env)
    return env


def train():
    monitor_dir = r'./monitor_log/ppo/'
    os.makedirs(monitor_dir, exist_ok=True)
    training_envs = DummyVecEnv([lambda : create_env() for _ in range(3)])
    model = PPO("MlpPolicy", training_envs, tensorboard_log="./tlog/ppo/", verbose=1, batch_size= 512)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=monitor_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    model.set_parameters(monitor_dir + "rl_model_1980000_steps.zip")
    model.learn(total_timesteps=200_0000, callback=checkpoint_callback)
    # model.learn(total_timesteps=200_0000)


def RRNtrain():
    monitor_dir = r'./monitor_log/rnnppo/'
    os.makedirs(monitor_dir, exist_ok=True)
    training_envs = DummyVecEnv([lambda : create_rnn_env() for _ in range(3)])
    model = RecurrentPPO("MlpLstmPolicy", training_envs,
                         batch_size=512,
                         # n_steps=128,
                         # n_epochs=10,
                         # policy_kwargs={'enable_critic_lstm': False, 'lstm_hidden_size': 128},
                         tensorboard_log="./tlog/rnnppo/",
                         verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=monitor_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    # model.set_parameters(monitor_dir + "rl_model_1980000_steps.zip")
    model.learn(total_timesteps=200_0000, callback=checkpoint_callback)
    # model.learn(total_timesteps=200_0000)



if __name__ == '__main__':
    train()
    #RRNtrain()








