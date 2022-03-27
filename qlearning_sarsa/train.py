import argparse
import wandb
import random
import numpy as np

from maze_env import MazeEnv
from model import Q_learning, Sarsa

parser = argparse.ArgumentParser(description='train q-learning sarsa')
parser.add_argument('--timesteps', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--sarsa', action='store_true')

arglist = parser.parse_args()

if __name__ == '__main__':
    env = MazeEnv(filename="./map.txt") # "./test-map.txt"

    random.seed(arglist.seed)
    np.random.seed(arglist.seed)

    method = Sarsa if arglist.sarsa else Q_learning
    model = method(env=env, alpha=arglist.alpha, gamma=arglist.gamma, seed=arglist.seed)
    wandb.init(name=model.model_name, project="qlearning-sarsa")
    model.learn(arglist.timesteps)


'''
cd ./qlearning_sarsa

python train.py --timesteps 50000 --alpha 0.02 --gamma 0.98 --seed 0
python train.py --timesteps 50000 --alpha 0.02 --gamma 0.98 --seed 42
python train.py --timesteps 50000 --alpha 0.02 --gamma 0.98 --seed 24

python train.py --timesteps 50000 --alpha 0.02 --gamma 0.98 --seed 0 --sarsa
python train.py --timesteps 50000 --alpha 0.02 --gamma 0.98 --seed 42 --sarsa
python train.py --timesteps 50000 --alpha 0.02 --gamma 0.98 --seed 24 --sarsa
'''