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


import ray
from ray import tune, air
from ray.rllib.utils.test_utils import check_learning_achieved

def train():
    ray.init(num_cpus=4)

    configs = {
        "PPO": {
            "num_sgd_iter": 5,
            "model": {
                "vf_share_layers": True,
            },
            "vf_loss_coeff": 0.0001,
        },
        "IMPALA": {
            "num_workers": 2,
            "num_gpus": 0,
            "vf_loss_coeff": 0.01,
        },
    }

    config = dict(
        configs["PPO"],
        **{
            "env": "TradingEnv2",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "use_lstm": True,
                "lstm_cell_size": 256,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
            },
            "framework": "torch",
            "_enable_learner_api": False,
            "_enable_rl_module_api": False,
        }
    )

    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": 5,
    }

    tuner = tune.Tuner(
        "PPO", param_space=config, run_config=air.RunConfig(stop=stop, verbose=2)
    )

    results = tuner.fit()

    ckpt = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint

    print(ckpt)


def train2():
    pass

if __name__ == '__main__':
    train()
