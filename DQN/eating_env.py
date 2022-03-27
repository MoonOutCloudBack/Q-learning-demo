import msilib
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

eating_map_size = 4

def generate_map(e_rate=0.3, map_size = eating_map_size):
    for i in range(100):
        chosen_x, chosen_y = np.random.randint(0, map_size-1), np.random.randint(0, map_size-1)
        char_list = []
        for x in range(map_size):
            string = ''
            for y in range(map_size):
                string += 'E' if (x == chosen_x and y == chosen_y) else '.'
            char_list.append(string+'\n')

        with open('./map/' + str(i) + '.txt', 'w') as f:
            f.writelines(char_list)

class EatingEnv:
    def __init__(self):
        self.map_size = eating_map_size
        self.map = np.zeros(shape=(2)) # self.map_size, self.map_size
        self.agent_pos = np.zeros(shape=(2))
        self.t = 0
        self.max_t = 40
        self.all_action_list = [
            np.array([0, 1]), np.array([0, -1]), 
            np.array([1, 0]), np.array([-1, 0]), 
            np.array([0, 0]), 
        ]
        self.reset()

    def read_map(self, filename):
        with open(filename, 'r') as f:
            world_data = [x for x in f.readlines()]
            self.map = np.zeros(shape=(2)) # self.map_size, self.map_size
            for i, line in enumerate(world_data):
                line = line.strip()
                for j, ch in enumerate(line):
                    if ch == 'E':
                        self.map[0], self.map[1] = i, j
        
    def reset(self):
        self.read_map(filename='./map/' + str(random.randint(0, 99)) + '.txt')
        self.t = 0
        self.agent_pos = np.array([random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)])
        # self.map[self.agent_pos[0]][self.agent_pos[1]] = 0
        return self._get_obs()
    
    def _get_obs(self):
        temp_map = np.concatenate((self.map.copy(), self.agent_pos.copy()), axis=0)
        # temp_map = np.expand_dims(temp_map, axis=0)
        return temp_map

    def step(self, action):
        self.t += 1
        reward = 0
        wanna_pos = np.array([self.agent_pos[0] + action[0], self.agent_pos[1] + action[1]])
        # not hit the wall
        if not (wanna_pos[0] < 0 or wanna_pos[0] >= self.map_size \
                or wanna_pos[1] < 0 or wanna_pos[1] >= self.map_size):
            self.agent_pos = wanna_pos
            # reach an E
            if wanna_pos[0] == self.map[0] and wanna_pos[1] == self.map[1]:
                reward = 1
                # self.map[wanna_pos[0]][wanna_pos[1]] = 0

        obs = self._get_obs()
        done = (np.sum(self.map) == 0) or (self.t >= self.max_t)
        info = {
            'success': (reward == 1),
            'timeout': (self.t >= self.max_t),
        }
        return obs, reward, done, info



if __name__ == '__main__':
    generate_map()