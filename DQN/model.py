import random
from tempfile import tempdir
import numpy as np
import wandb
import time

import copy      
import os

import torch
from torch._C import device
from torch.nn import functional as F
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal 

from eating_env import EatingEnv, action_to_encoding, encoding_to_action


class RolloutBuffer():
    def __init__(self, map_size, size=int(1e5)):
        self.map_size = map_size
        self.size = size
        self.index = 0
        self.max_index = 0
        self.obs_buffer = torch.zeros(size, 4)
        self.action_buffer = torch.zeros(size, 2)
        self.reward_buffer = torch.zeros(size, 1)
        self.obsnew_buffer = torch.zeros(size, 4)

    def add(self, samples):
        for sample in samples:
            self.obs_buffer[self.index] = torch.tensor(sample[0])
            self.action_buffer[self.index] = torch.tensor(sample[1])
            self.reward_buffer[self.index] = torch.tensor(sample[2])
            self.obsnew_buffer[self.index] = torch.tensor(sample[3])
            self.index = (self.index + 1) % self.size
            self.max_index = min(self.size, self.index)

    def sample(self, k:int):
        transitions = []
        indexes = [random.randint(0, self.max_index) for i in range(k)]
        transitions = (
            self.obs_buffer[indexes].clone(), self.action_buffer[indexes].clone(),
            self.reward_buffer[indexes].clone(), self.obsnew_buffer[indexes].clone(),
        )
        return transitions


class DQN(nn.Module):
    def __init__(self, env: EatingEnv, learning_rate, gamma=0.95, use_target_network=False, seed=0):
        super(DQN, self).__init__()
        self.env = env
        self.gamma = gamma
        self.use_target_network = use_target_network
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.model_name = 'DQN-' + str(self.gamma) + '-' + str(learning_rate) # time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        if self.use_target_network:
            self.model_name += '-target'
        
        self.eval_frequency = 1
        self.target_sync_frequency = 20 if self.use_target_network else 1
        self.eval_steps = 500
        self.epsilon_greedy = 0.2
        self.sample_times = 100
        self.update_times = 3
        self.batch_size = 256

        self.value_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.target_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).eval()
        self.state = {'value_net': self.value_net.state_dict()}

        self.optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)
        self.rollout_buffer = RolloutBuffer(map_size=self.env.map_size)


    def forward(self, obs, action, use_batch=True):
        obs = torch.tensor(obs).to(torch.float32)
        action = torch.tensor(action).to(torch.float32)
        if not use_batch:
            obs = obs.unsqueeze(dim=0)
            action = action.unsqueeze(dim=0)

        value = self.value_net(torch.concat((obs, action), dim=1))
        value = value.squeeze(dim=1)

        if not use_batch:
            value = value.squeeze(dim=0).item()

        return value


    def forward_target(self, obs, action, use_batch=True):
        obs = torch.tensor(obs).to(torch.float32)
        action = torch.tensor(action).to(torch.float32)
        if not use_batch:
            obs = obs.unsqueeze(dim=0)
            action = action.unsqueeze(dim=0)

        target = self.target_net(torch.concat((obs, action), dim=1))
        target = target.squeeze(dim=1)

        if not use_batch:
            target = target.squeeze(dim=0).item()

        return target


    def learn(self, timesteps: int):
        for i in range(10):
            self.rollout_buffer.add(self.collect_rollout())

        for i in range(timesteps):
            print("learn steps =", i)
            self.avg_loss = self.train()
            if i % self.eval_frequency == 0:
                self.evaluate()
            if i % self.target_sync_frequency == 0: # sync target net
                self.target_net.load_state_dict(self.value_net.state_dict())
        self.save(file=self.model_name)


    def train(self):
        self.rollout_buffer.add(self.collect_rollout()) # off-policy, epsilon-greedy sampling
        value_losses = []

        for i in range(self.update_times):
            print("update step =", i)
            trainsitions = self.rollout_buffer.sample(k=self.batch_size)
            loss = self.get_loss(trainsitions)
            value_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return sum(value_losses)/len(value_losses)


    def get_loss(self, transitions):
        obs, action, reward, obsnew = transitions
        q = self.forward(obs, action)
        q_prime = self.get_best_action_q(obsnew, batch_size=self.batch_size)
        y = reward + self.gamma * q_prime
        y = y.squeeze(dim=1)
        loss = F.mse_loss(q, y)
        return loss


    def get_best_action_q(self, obs, batch_size):
        temp_q = torch.zeros(batch_size, 5)
        for i, action in enumerate(self.env.all_action_list):
            action_batch = torch.tensor(action).unsqueeze(dim=0).repeat_interleave(batch_size, dim=0)
            temp_q[:, i] = self.forward_target(obs, action_batch, use_batch=True)

        max_q, action_indexes = torch.max(temp_q, dim=1, keepdim=True)
        return max_q


    def policy(self, obs, training=False):
        if training and random.random() < self.epsilon_greedy:
            return random.choice(self.env.all_action_list)

        temp_q = np.zeros((5))
        for i, action in enumerate(self.env.all_action_list):
            temp_q[i] = self.forward(obs, action, use_batch=False)

        max_index = random.choice(np.where(temp_q == np.max(temp_q))[0])
        action = self.env.all_action_list[max_index]

        if random.random() < 0.001:
            print("policy action =", action)
        return action


    def collect_rollout(self):
        rollout = []
        for _ in range(self.sample_times):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.policy(obs, training=True)
                obs_new, reward, done, info = self.env.step(action)
                rollout.append([obs, action, reward, obs_new])
                if info['success']:
                    for _ in range(5):
                        rollout.append([obs, action, reward, obs_new])
                obs = obs_new
        return rollout


    def evaluate(self):
        success_rate = []
        timeout_rate = []
        success_steps = []
        avg_reward = []

        rewards = 0
        obs = self.env.reset()
        for _ in range(self.eval_steps):
            action = self.policy(obs)
            obs, reward, done, info = self.env.step(action)
            rewards += reward
            if done:
                success_rate.append(1 if info['success'] else 0)
                timeout_rate.append(1 if info['timeout'] else 0)
                if info['success']:
                    success_steps.append(self.env.t)
                avg_reward.append(rewards)

                rewards = 0
                obs = self.env.reset()

        max_value = -100
        for action in self.env.all_action_list:
            for x in range(self.env.map_size):
                for y in range(self.env.map_size):
                    for i in range(self.env.map_size):
                        for j in range(self.env.map_size):
                            value = self.forward(np.array([x, y, i, j]), action, use_batch=False)
                            max_value = max(max_value, value)


        wandb.log({
            'max q value': max_value,
            'successful rate': 0 if len(success_rate) == 0 else np.mean(np.array(success_rate)),
            'timeout rate': 0 if len(timeout_rate) == 0 else np.mean(np.array(timeout_rate)),
            'average total rewards': 0 if len(avg_reward) == 0 else np.mean(np.array(avg_reward)),
            'average finish steps': 0 if len(success_steps) == 0 else np.mean(np.array(success_steps)),
            'average loss': self.avg_loss,
        })


    def load(self, file):
        checkpoint = torch.load(file)
        # self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        # self.action_encoder.load_state_dict(checkpoint['action_encoder'])
        self.value_net.load_state_dict(checkpoint['value_net'])


    def save(self, file):
        torch.save(self.state, file)




