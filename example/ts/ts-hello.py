import gymnasium as gym
import tianshou as ts

env = gym.make('CartPole-v1')


import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

# result = ts.trainer.offpolicy_trainer(
#     policy, train_collector, test_collector,
#     max_epoch=10, step_per_epoch=10000, step_per_collect=10,
#     update_per_step=0.1, episode_per_test=100, batch_size=64,
#     train_fn=lambda epoch, env_step: policy.set_eps(0.1),
#     test_fn=lambda epoch, env_step: policy.set_eps(0.05),
#     stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
# print(f'Finished training! Use {result["duration"]}')

# pre-collect at least 5000 transitions with random action before training
train_collector.collect(n_step=5000, random=True)

policy.set_eps(0.1)
for i in range(int(1e6)):  # total step
    collect_result = train_collector.collect(n_step=10)

    # once if the collected episodes' mean returns reach the threshold,
    # or every 1000 steps, we test it on test_collector
    if collect_result['rews'].mean() >= env.spec.reward_threshold or i % 1000 == 0:
        policy.set_eps(0.05)
        result = test_collector.collect(n_episode=100)
        if result['rews'].mean() >= env.spec.reward_threshold:
            print(f'Finished training! Test mean returns: {result["rews"].mean()}')
            break
        else:
            # back to training eps
            policy.set_eps(0.1)

    # train policy with a sampled batch data from buffer
    losses = policy.update(64, train_collector.buffer)

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
writer = SummaryWriter('log/dqn')
logger = TensorboardLogger(writer)

# torch.save(policy.state_dict(), 'dqn.pth')
# policy.load_state_dict(torch.load('dqn.pth'))