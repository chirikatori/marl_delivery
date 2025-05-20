import torch
from torch import nn, optim
from collections import deque
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, obs_size, n_robots, n_actions, network=None):
        super(DQN, self).__init__()
        self.n_robots = n_robots
        self.n_actions = n_actions

        if network:
            self.net = network
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_robots * n_actions),
            )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array,
                                                           zip(*samples))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_size, n_robots, lr=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=64,
                 replay_buffer_capacity=10000):
        self.obs_size = obs_size
        self.n_robots = n_robots
        self.n_actions = 5 * 3
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa

        self.policy_net = DQN(obs_size, n_robots, self.n_actions).to(self.device)  # noqa
        self.target_net = DQN(obs_size, n_robots, self.n_actions).to(self.device)  # noqa
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            actions = np.array([
                [random.randint(0, 4), random.randint(0, 2)]
                for _ in range(self.n_robots)
            ])
            return actions
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_values = q_values.view(self.n_robots, -1)

                actions = []
                for q in q_values:
                    action = torch.argmax(q).item()
                    actions.append([action // 3, action % 3])
                actions = np.array(actions)
                return actions

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        action_id = action[..., 0] * 3 + action[..., 1]

        q_values = self.policy_net(state)
        q_values = q_values.view(self.batch_size, self.n_robots, -1)
        q_values = q_values.gather(2, action_id.unsqueeze(-1)).squeeze(-1)
        q_values = q_values.mean(1, keepdim=True)

        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_state).view(
                self.batch_size, self.n_robots, -1)
            next_actions = next_q_values_policy.argmax(2)

            next_q_values_target = self.target_net(next_state).view(
                self.batch_size, self.n_robots, -1)
            next_q_values = next_q_values_target.gather(
                2, next_actions.unsqueeze(-1)).squeeze(-1)
            next_q_values = next_q_values.mean(1, keepdim=True)

            target_q = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def buffer_push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
