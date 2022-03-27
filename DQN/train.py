import argparse
import wandb
import random
import numpy as np

from eating_env import EatingEnv
from model import DQN

parser = argparse.ArgumentParser(description='train q-learning sarsa')
parser.add_argument('--timesteps', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--target_network', action='store_true')
parser.add_argument('--seed', type=int, default=0, required=False)

arglist = parser.parse_args()

if __name__ == '__main__':
    env = EatingEnv()

    random.seed(arglist.seed)
    np.random.seed(arglist.seed)

    model = DQN(env=env, learning_rate=arglist.learning_rate, gamma=arglist.gamma, 
        use_target_network=arglist.target_network, seed=arglist.seed)
    wandb.init(name=model.model_name, project="DQN")
    model.learn(arglist.timesteps)


'''
cd ./DQN
python eating_env.py

python train.py --timesteps 600 --learning_rate 0.001 --gamma 0.98 
python train.py --timesteps 600 --learning_rate 0.001 --gamma 0.98 --target_network
'''