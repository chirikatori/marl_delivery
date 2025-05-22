import gymnasium as gym
from gymnasium import spaces

from env import Environment
import numpy as np


def convert_state(state):
    ret_state = {}
    ret_state["robots"] = np.array(state["robots"]).astype(np.float32).flatten()
    ret_state["packages"] = np.array(state["packages"]).astype(np.float32).flatten()[:100]
    if len(ret_state["packages"]) < 100:
        ret_state["packages"] = np.concatenate((ret_state["packages"], np.zeros(100 - len(ret_state["packages"]))))
    return np.concatenate(list(ret_state.values()))

def reward_shaping(r, env, state, action):
    return r

class Env(gym.Env):
    def __init__(self, *args, **kwargs):
        super(Env, self).__init__()
        self.env = Environment(*args, **kwargs)
        self.action_space = spaces.multi_discrete.MultiDiscrete([5, 3] * self.env.n_robots)
        self.prev_state = self.env.reset()
        first_state = convert_state(self.prev_state)
        self.observation_space = spaces.Box(low=0, high=100, shape=first_state.shape, dtype=np.float32)
        from sklearn.preprocessing import LabelEncoder
        self.le1, self.le2 = LabelEncoder(), LabelEncoder()
        self.le1.fit(['S', 'L', 'R', 'U', 'D'])
        self.le2.fit(['0', '1', '2'])

    def reset(self, *args, **kwargs):
        self.prev_state = self.env.reset()
        return convert_state(self.prev_state), {}

    def render(self, *args, **kwargs):
        return self.env.render()

    def step(self, action):
        moves = self.le1.inverse_transform(action.reshape(-1, 2).T[0])
        pkgs = self.le2.inverse_transform(action.reshape(-1, 2).T[1])
        action = list(zip(moves, pkgs))
        print(len(action))
        s, r, done, infos = self.env.step(action)
        new_r = reward_shaping(r, self.env, self.prev_state, action)
        self.prev_state = s
        return convert_state(s), new_r, done, False, infos
