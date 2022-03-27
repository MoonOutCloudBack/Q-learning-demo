import random
import numpy as np
import wandb

from maze_env import MazeEnv, action_to_encoding, encoding_to_action

class TD_tabular:
    def __init__(self, env: MazeEnv, alpha, gamma, seed):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.model_name = 'base_class'
        self.eval_frequency = 10
        self.eval_steps = 5000
        self.epsilon_greedy = 0.2
        self.sample_times = 50
        self.update_times = 200
        self.q_function = np.zeros(shape=(env.map_size, env.map_size, len(env.all_action_list)))

    def learn(self, timesteps: int):
        for i in range(timesteps):
            self.train()
            if i % self.eval_frequency == 0:
                self.evaluate()
        self.save('./' + self.model_name + '-qf.npy')

    def train(self):
        raise NotImplementedError()

    def save(self, save_path: str = "./qf.npy"):
        np.save(save_path, self.q_function)

    def policy(self, obs, training=False):
        if training and random.random() < self.epsilon_greedy:
            return random.choice(self.env.all_action_list)
        temp_q = self.q_function[obs[0]][obs[1]]
        encoding = random.choice(np.where(temp_q == np.max(temp_q))[0])
        action = encoding_to_action[encoding]
        return action

    def collect_rollout(self):
        rollout = []
        for _ in range(self.sample_times):
            obs = self.env.reset()
            done = False
            episode = []
            while not done:
                action = self.policy(obs, training=True)
                obs_new, reward, done, info = self.env.step(action)
                episode.append([obs, action, reward])
                obs = obs_new
            rollout.append(episode)
        return rollout

    def evaluate(self):
        success_rate = []
        fail_rate = []
        timeout_rate = []
        success_steps = []
        avg_reward = []
        performance = []

        rewards = 0
        obs = self.env.reset()
        for _ in range(self.eval_steps):
            action = self.policy(obs)
            obs, reward, done, info = self.env.step(action)
            rewards += reward
            if done:
                success_rate.append(1 if info['success'] else 0)
                fail_rate.append(1 if info['fail'] else 0)
                timeout_rate.append(1 if info['timeout'] else 0)
                if info['success']:
                    success_steps.append(self.env.t)
                avg_reward.append(rewards)
                performance.append(1/(self.env.t-18) if info['success'] else 0)

                rewards = 0
                obs = self.env.reset()

        wandb.log({
            'performance': 0 if len(performance) == 0 else np.mean(np.array(performance)),
            'successful rate': 0 if len(success_rate) == 0 else np.mean(np.array(success_rate)),
            'fail rate': 0 if len(fail_rate) == 0 else np.mean(np.array(fail_rate)),
            'timeout rate': 0 if len(timeout_rate) == 0 else np.mean(np.array(timeout_rate)),
            'average total rewards': 0 if len(avg_reward) == 0 else np.mean(np.array(avg_reward)),
            'average finish steps': 0 if len(success_steps) == 0 else np.mean(np.array(success_steps)),
        })


class Q_learning(TD_tabular):
    def __init__(self, env: MazeEnv, alpha: float=0.05, gamma: float=0.98, seed=0):
        TD_tabular.__init__(self, env, alpha, gamma, seed)
        self.model_name = 'qlearning-' + str(alpha) + '-' + str(gamma) + '-' + str(seed)
        self.rollout_buffer = []

    def train(self):
        self.rollout_buffer += self.collect_rollout() # off-policy, epsilon-greedy sampling
        for _ in range(self.update_times):
            episode = random.choice(self.rollout_buffer)
            epi_len = len(episode)
            index = random.randint(0, epi_len-1)
            obs, encoding, reward = episode[index][0], \
                action_to_encoding[episode[index][1]], episode[index][2]
            if index == epi_len-1:
                self.q_function[obs[0]][obs[1]][encoding] += self.alpha * \
                    (reward - self.q_function[obs[0]][obs[1]][encoding])
            else: 
                next_obs = episode[index+1][0]
                next_encoding = action_to_encoding[self.policy(next_obs)] # argmax of current policy
                self.q_function[obs[0]][obs[1]][encoding] += self.alpha * \
                    (reward + self.gamma * self.q_function[next_obs[0]][next_obs[1]][next_encoding] \
                        - self.q_function[obs[0]][obs[1]][encoding])


class Sarsa(TD_tabular):
    def __init__(self, env: MazeEnv, alpha: float = 0.05, gamma: float=0.98, seed=0):
        TD_tabular.__init__(self, env, alpha, gamma, seed)
        self.model_name = 'sarsa-' + str(alpha) + '-' + str(gamma) + '-' + str(seed)

    def train(self):
        rollout = self.collect_rollout()
        for _ in range(self.update_times):
            episode = random.choice(rollout) # on-policy
            epi_len = len(episode)
            index = random.randint(0, epi_len-1)
            obs, encoding, reward = episode[index][0], \
                action_to_encoding[episode[index][1]], episode[index][2]
            if index == epi_len-1:
                self.q_function[obs[0]][obs[1]][encoding] += self.alpha * \
                    (reward - self.q_function[obs[0]][obs[1]][encoding])
            else: 
                next_obs, next_encoding = episode[index+1][0], \
                    action_to_encoding[episode[index+1][1]] # a_{t+1} in trajectory
                self.q_function[obs[0]][obs[1]][encoding] += self.alpha * \
                    (reward + self.gamma * self.q_function[next_obs[0]][next_obs[1]][next_encoding] \
                        - self.q_function[obs[0]][obs[1]][encoding])

