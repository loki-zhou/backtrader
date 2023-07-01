import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib import QRDQN
from pandas_ta.statistics import zscore
import legendary_ta as lta

windows_size = 15
# Import your datas
# df = pd.read_csv("data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col= "date")
# df.sort_index(inplace= True)
# df.dropna(inplace= True)
# df.drop_duplicates(inplace=True)

#df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

# Generating features
# WARNING : the column names need to contain keyword 'feature' !
# df["feature_raw_close"] = df["close"]
# df["feature_raw_open"] = df["open"]
# df["feature_raw_low"] = df["low"]
# df["feature_raw_high"] = df["high"]
# df["feature_raw_volume"] = df["volume"]
# lta.pinbar(df)

df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"]/df["close"]
df["feature_low"] = df["low"]/df["close"]
df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()

# df['feature_z_close'] = zscore(df['close'], length=30 )
# df['feature_z_open'] = zscore(df['open'], length=30 )
# df['feature_z_high'] = zscore(df['high'], length=30 )
# df['feature_z_low'] = zscore(df['low'], length=30 )
# df['feature_z_volume'] = zscore(df['volume'], length=30 )
# CustomStrategy = ta.Strategy(
#     name="Momo and Volatility",
#     description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
#     ta=[
#         {"kind": "sma", "length": 20,"prefix": "feature"},
#         {"kind": "sma", "length": 50,"prefix": "feature"},
#         {"kind": "sma", "length": 200, "prefix": "feature"},
#         {"kind": "bbands", "length": 20, "prefix": "feature"},
#         {"kind": "rsi", "prefix": "feature"},
#         {"kind": "macd", "fast": 8, "slow": 21, "prefix": "feature"},
#         {"kind": "sma", "close": "volume", "length": 20, "prefix": "feature_VOLUME"},
#     ]
# )
# df.ta.cores = 0
# df.ta.strategy(CustomStrategy)
df.dropna(inplace= True)


# Create your own reward function with the history object
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
    env = gym.make(
            "TradingEnv",
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
    env.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.add_metric('Max Drawdown', max_drawdown)
    env = Monitor(env, monitor_dir)
    env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.NormalizeReward(env)
    return env
monitor_dir = r'./monitor_log/ppo/'
os.makedirs(monitor_dir, exist_ok=True)

def train():
    env = create_env()
    model = PPO("MlpPolicy", env, tensorboard_log="./tlog/ppo/", verbose=1, batch_size=256)
    #model = QRDQN("MlpPolicy", env, verbose=1)
    # model = RecurrentPPO("MlpLstmPolicy", env,
    #                      # batch_size=256,
    #                      # n_steps=128,
    #                      # n_epochs=10,
    #                      # policy_kwargs={'enable_critic_lstm': False, 'lstm_hidden_size': 128},
    #                      verbose=1)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=monitor_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=monitor_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    model.set_parameters(monitor_dir + "rl_model_2000000_steps.zip")
    model.learn(total_timesteps=200_0000, callback=checkpoint_callback)

def test():
    #model = QRDQN.load(monitor_dir + "rl_model_500000_steps.zip")
    model = PPO.load(monitor_dir + "rl_model_2000000_steps.zip")
    #model = RecurrentPPO.load(monitor_dir + "rl_model_2000000_steps.zip")
    env = create_env()
    done, truncated = False, False
    observation, info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while not done and not truncated:
        action, _states = model.predict(observation)
        #action = env.action_space.sample()
        #action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_starts, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
    env.save_for_render(dir="./render_logs")

# tensorboard.exe --logdir example/gym/tlog/ppo
if __name__ == '__main__':
    #train()
    test()