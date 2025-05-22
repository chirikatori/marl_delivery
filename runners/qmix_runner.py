import torch
import numpy as np
import wandb

from algorithms.replay_buffer import ReplayBufferQMix as ReplayBuffer
from algorithms import QMIX
from algorithms.utils import Normalization

from config import Env_Args, QMIX_Args
from env import Environment 


class Runner_QMIX:
    def __init__(self, args: QMIX_Args, env_args: Env_Args, seed):
        self.args = args
        self.env_args = env_args
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize Weights & Biases
        wandb.init(
            project="qmix_marl_delivery",
            config={**vars(args), **vars(env_args), "seed": seed},
        )

        # Create custom grid-world env
        self.env = Environment(
            map_file=env_args.map_file,
            max_time_steps=env_args.episode_limit,
            n_robots=env_args.n_robots,
            n_packages=env_args.n_packages,
            seed=self.seed,
        )

        # dims for QMIX
        self.args.obs_dim = 3
        self.args.state_dim = (
            self.env.n_rows * self.env.n_cols
            + self.env.n_robots * 3
            + self.env.n_packages * 4
        )
        self.args.action_dim = 5 * 3  # 5 moves × 3 pkg actions

        # instantiate agent and buffer
        self.agent_n = QMIX(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.epsilon = args.epsilon
        self.total_steps = 0
        if args.use_reward_norm:
            self.reward_norm = Normalization(shape=1)

    def run(self):
        eval_count = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > eval_count:
                self.evaluate_policy()
                eval_count += 1

            _, ep_reward, ep_len = self.run_episode(evaluate=False)
            self.total_steps += ep_len

            # log episode reward
            wandb.log({"episode_reward": ep_reward, "step": self.total_steps})

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)

        # final evaluation
        self.evaluate_policy()
        wandb.finish()

    def run_episode(self, evaluate=False):
        episode_reward = 0.0
        self.env.reset()

        if self.args.use_rnn:
            self.agent_n.eval_Q_net.rnn_hidden = None

        last_onehot = np.zeros((self.args.N, self.args.action_dim), dtype=np.float32)

        for t in range(self.args.episode_limit):
            state_dict = self.env.get_state()
            state = self._encode_state(state_dict)
            obs_n = self._encode_obs()
            avail_a = np.ones((self.args.N, 5*3), dtype=np.float32)

            eps = 0.0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot, avail_a, eps)
            actions = [(self._move_from_idx(a // 3), str(a % 3)) for a in a_n]

            _, r, done, infos = self.env.step(actions)
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                dw = done and (t + 1) != self.args.episode_limit
                self.replay_buffer.store_transition(
                    t, obs_n, state, avail_a, last_onehot, a_n, r, dw
                )
                self.epsilon = max(
                    self.args.epsilon_min,
                    self.epsilon - self.args.epsilon_decay
                )

            last_onehot = np.eye(self.args.action_dim)[a_n]
            if done:
                break

        if not evaluate:
            final_obs_n = self._encode_obs()
            final_state = self._encode_state(self.env.get_state())
            final_avail = np.ones((self.args.N, 5*3), dtype=np.float32)
            self.replay_buffer.store_last_step(t + 1, final_obs_n, final_state, final_avail)

        return False, episode_reward, t + 1

    def evaluate_policy(self):
        total_reward = 0.0
        for _ in range(self.args.evaluate_times):
            _, r, _ = self.run_episode(evaluate=True)
            total_reward += r
        avg_reward = total_reward / self.args.evaluate_times
        print(f"[Eval] steps={self.total_steps}, avg_reward={avg_reward:.2f}")
        wandb.log({"eval_avg_reward": avg_reward, "step": self.total_steps})

    def _encode_obs(self):
        obs = []
        for robot in self.env.robots:
            r, c = robot.position
            obs.append(np.array([r, c, robot.carrying], dtype=np.float32))
        return np.stack(obs, axis=0)

    def _encode_state(self, state_dict):
        grid = np.array(state_dict["map"], dtype=np.float32).flatten()
        robots = []
        for (r, c, carry) in state_dict["robots"]:
            robots.extend([r - 1, c - 1, carry])
        robots = np.array(robots, dtype=np.float32)

        pkgs = []
        for (_, sr, sc, tr, tc, _, _) in state_dict["packages"]:
            pkgs.extend([sr - 1, sc - 1, tr - 1, tc - 1])
        pad_len = self.env_args.n_packages * 4 - len(pkgs)
        pkgs = np.array(pkgs + [0] * pad_len, dtype=np.float32)

        return np.concatenate([grid, robots, pkgs], axis=0)

    def _move_from_idx(self, idx):
        return ['S', 'L', 'R', 'U', 'D'][idx]
