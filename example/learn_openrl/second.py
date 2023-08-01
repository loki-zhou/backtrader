# train_ppo.py
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

def train():
    # 创建 MPE 环境，使用异步环境，即每个智能体独立运行
    env = make(
        "simple_spread",
        env_num=100,
        asynchronous=True,
    )
    # 创建 神经网络，使用GPU进行训练
    net = Net(env, device="cuda")
    agent = Agent(net) # 初始化训练器
    # 开始训练
    agent.train(total_time_steps=5000000)
    # 保存训练完成的智能体
    agent.save("./ppo_agent/")
if __name__ == "__main__":
    train()