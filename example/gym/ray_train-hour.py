import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
import custom_indicator as cta

def load_data():
    df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    # df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    # df.sort_index(inplace=True)
    # df.dropna(inplace=True)
    # df.drop_duplicates(inplace=True)
    # df["feature_close"] = df["close"].pct_change()
    # df["feature_open"] = df["open"] / df["close"]
    # df["feature_high"] = df["high"] / df["close"]
    # df["feature_low"] = df["low"] / df["close"]
    # df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    cta.NormalizedScore(df, 30)
    df.dropna(inplace=True)
    return df

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

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



def create_env(config):
    df = load_data()
    env = TradingEnv(
        name="BTCUSD",
        df=df,
        windows=1,
        positions=[-1, -0.5, 0, 0.5, 1],  # From -1 (=SHORT), to +1 (=LONG)
        # initial_position = 'random', #Initial position
        initial_position=0,  # Initial position
        trading_fees=0.1 / 100,  # 0.01% per stock buy / sell
        borrow_interest_rate=0,  # per timestep (= 1h here)
        reward_function=reward_function,
        portfolio_initial_value=10000,  # in FIAT (here, USD)
        # max_episode_duration = 2400,
        # max_episode_duration=500,
    )
    env.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.add_metric('Max Drawdown', max_drawdown)
    env = gym.wrappers.NormalizeObservation(env)
    return env

from ray.tune.registry import register_env
register_env("TradingEnv2", create_env)


from ray.rllib.algorithms.ppo import PPOConfig

# stop = {
#     "training_iteration": args.stop_iters,
#     "timesteps_total": args.stop_timesteps,
#     "episode_reward_mean": args.stop_reward,
# }

stop = {
    #"episode_reward_min": 500,
    "episode_reward_mean": 5,
}
import ray
from ray import tune, air

def train():
    ray.init(num_cpus=4)


    config = PPOConfig().environment(env="TradingEnv2",env_config = {
    }).training(
        gamma=0.99,
        entropy_coeff=0.001,
        num_sgd_iter=10,
        vf_loss_coeff=1e-5,
        model={
            "use_attention": True,
            "max_seq_len": 10,
            "attention_num_transformer_units": 1,
            "attention_dim": 32,
            "attention_memory_inference": 10,
            "attention_memory_training": 10,
            "attention_num_heads": 1,
            "attention_head_dim": 32,
            "attention_position_wise_mlp_dim": 32,
        },
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"episode_reward_mean": 5},
            checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
        ),
        param_space=config,

    )

    results = tuner.fit()

    ckpt = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint

    print(ckpt)


def train2():
    pass

if __name__ == '__main__':
    train()
