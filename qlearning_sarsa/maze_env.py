import sys
import random
import numpy as np

action_to_encoding = {
    (0, 0): 0,
    (0, 1): 1,
    (0, -1): 2,
    (1, 0): 3,
    (-1, 0): 4,
}
encoding_to_action ={
    0: (0, 0), 
    1: (0, 1), 
    2: (0, -1), 
    3: (1, 0), 
    4: (-1, 0), 
}

class MazeEnv:
    def __init__(self, filename: str = "./map.txt"):
        self.map = []
        self.map_size = -1
        self.agent_pos = (-1, -1)
        self.read_map(filename)

        self.t = 0
        self.max_t = 500
        self.all_action_list = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        self.is_visited = np.zeros(shape=(self.map_size, self.map_size))
        self.reset()

    def read_map(self, filename):
        with open(filename, 'r') as f:
            world_data = [x for x in f.readlines()]
            self.map = []
            for i, line in enumerate(world_data):
                if i == 0:
                    self.map_size = int(line)
                else:
                    line = line.strip()
                    self.map.append([x for x in line])
        
        self.start_pos_list, self.finish_pos_list = [], []
        for x, row in enumerate(self.map):
            for y, ch in enumerate(row):
                if ch == 'S':
                    self.start_pos_list.append((x, y))
                if ch == 'F':
                    self.finish_pos_list.append((x, y))

    def reset(self):
        self.t = 0
        self.agent_pos = random.choice(self.start_pos_list)
        self.is_visited = np.zeros(shape=(self.map_size, self.map_size))
        self.is_visited[self.agent_pos[0]][self.agent_pos[1]] = 1
        return self._get_obs()
    
    def _get_obs(self):
        return self.agent_pos

    def step(self, action):
        self.t += 1
        reward = 0
        if action != (0, 0):
            wanna_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
            # hit the wall
            if wanna_pos[0] < 0 or wanna_pos[0] >= self.map_size \
                    or wanna_pos[1] < 0 or wanna_pos[1] >= self.map_size \
                    or self.map[wanna_pos[0]][wanna_pos[1]] == '#':
                reward = -1
            else:
                self.agent_pos = wanna_pos
                # reach the goal
                if self.map[wanna_pos[0]][wanna_pos[1]] == 'F':
                    reward = 1
                    for i in range(self.map_size):
                        for j in range(self.map_size):
                            if self.is_visited[i][j] == 0:
                                reward += 0.001
                # reach a new state
                elif self.is_visited[wanna_pos[0]][wanna_pos[1]] == 0:
                    self.is_visited[wanna_pos[0]][wanna_pos[1]] = 1
                    reward = 0.001

        obs = self._get_obs()
        done = (reward >= 1) or (reward == -1) or (self.t >= self.max_t)
        info = {
            'success': (reward >= 1),
            'fail': (reward == -1),
            'timeout': (self.t >= self.max_t),
        }
        return obs, reward, done, info

    def print_map(self):
        text = ''
        print()
        for line in map:
            for cell in line:
                text += cell
            text += '\n'
        sys.stdout.write('\r' + text)
        sys.stdout.flush()
