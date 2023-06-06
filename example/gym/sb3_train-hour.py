import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
from common import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


# Import your datas
df = pd.read_csv("data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

# Generating features
# WARNING : the column names need to contain keyword 'feature' !
df["feature_raw_close"] = df["close"]
df["feature_raw_open"] = df["open"]
df["feature_raw_low"] = df["low"]
df["feature_raw_high"] = df["high"]
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"]/df["close"]
df["feature_low"] = df["low"]/df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()
df.dropna(inplace= True)


# Create your own reward function with the history object
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

def create_env():
    env = gym.make(
            "TradingEnv",
            name= "BTCUSD",
            df = df,
            windows= 5,
            positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
            # initial_position = 'random', #Initial position
            initial_position=0,  # Initial position
            trading_fees = 0.01/100, # 0.01% per stock buy / sell
            borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
            reward_function = reward_function,
            portfolio_initial_value = 1000, # in FIAT (here, USD)
            max_episode_duration = 2000,
            #max_episode_duration=500,
        )
    return env
monitor_dir = r'./monitor_log/'
os.makedirs(monitor_dir, exist_ok=True)

def train():
    env = create_env()
    env = Monitor(env, monitor_dir)

    #model = PPO("MlpPolicy", env, verbose=1)
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=monitor_dir)
    model.learn(total_timesteps=1000_0000, callback=callback)

def test():
    model = PPO.load(monitor_dir + "best_model.zip")
    env = create_env()
    done, truncated = False, False
    observation, info = env.reset()
    lstm_states = None
    while not done and not truncated:
        #action, _states = model.predict(observation)
        #action = env.action_space.sample()
        action, lstm_states = model.predict(observation, state=lstm_states,  deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
    env.save_for_render(dir="./render_logs")

if __name__ == '__main__':
    train()
    #test()