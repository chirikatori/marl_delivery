import torch
import numpy as np
import wandb

from algorithms.utils import Normalization, RewardScaling
from algorithms.replay_buffer import ReplayBufferPPO as ReplayBuffer
from algorithms import MAPPO
from env import Environment  # use original Environment class
from config import MAPPO_Args, Env_Args

# helper to flatten state dict into vector
def convert_state(state, max_package_slots=100):
    robots = np.array(state['robots'], dtype=np.float32).flatten()
    packages = np.array(state['packages'], dtype=np.float32).flatten()
    if packages.size < max_package_slots:
        packages = np.pad(packages, (0, max_package_slots - packages.size), 'constant')
    else:
        packages = packages[:max_package_slots]
    return np.concatenate([robots, packages])

# map discrete indices back to actions
MOVES = ['S', 'L', 'R', 'U', 'D']
PKG_ACTIONS = ['0', '1', '2']

def decode_actions(flat_actions, n_agents):
    moves_idx, pkgs_idx = [], []
    for a in flat_actions:
        m, p = np.unravel_index(int(a), (len(MOVES), len(PKG_ACTIONS)))
        moves_idx.append(MOVES[m])
        pkgs_idx.append(PKG_ACTIONS[p])
    return list(zip(moves_idx, pkgs_idx))

class Runner_MAPPO:
    def __init__(self, args: MAPPO_Args, env_args: Env_Args, seed: int):
        self.args = args
        self.env_args = env_args
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize wandb
        wandb.init(project="marl_delivery", config={**vars(args), **vars(env_args), "seed": seed})

        # Initialize original Environment
        self.env = Environment(
            map_file=env_args.map_file,
            max_time_steps=env_args.episode_limit,
            n_robots=env_args.n_robots,
            n_packages=env_args.n_packages,
            seed=self.seed
        )

        # Setup obs and action dims
        self.args.N = self.env.n_robots
        init_state = self.env.reset()
        obs_vec = convert_state(init_state)
        obs_dim = obs_vec.shape[0]
        self.args.obs_dim = obs_dim
        self.args.state_dim = obs_dim
        self.args.obs_dim_n = [obs_dim] * self.args.N

        action_dim = len(MOVES) * len(PKG_ACTIONS)
        self.args.action_dim = action_dim
        self.args.action_dim_n = [action_dim] * self.args.N

        print(f"[ENV INFO] N_agents={self.args.N}, obs_dim_n={self.args.obs_dim_n}, action_dim_n={self.args.action_dim_n}")

        # Initialize MAPPO agent and replay buffer
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.total_steps = 0

        # Reward norm/scaling
        if self.args.use_reward_norm:
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        eval_count = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > eval_count:
                self.evaluate_policy()
                eval_count += 1

            ep_r, steps = self.run_episode(evaluate=False)
            self.total_steps += steps

            if self.replay_buffer.episode_num >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()

    def run_episode(self, evaluate=False):
        state = self.env.reset()
        obs_vec = convert_state(state)
        obs_n = [obs_vec] * self.args.N

        if getattr(self.args, 'use_reward_scaling', False) and not evaluate:
            self.reward_scaling.reset()
        if getattr(self.args, 'use_rnn', False):
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None

        episode_reward = 0.0
        for step in range(self.args.episode_limit):
            flat_actions, logprobs = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            v_n_current = self.agent_n.get_value(obs_n[0])
            decoded = decode_actions(flat_actions, self.args.N)

            next_state, r, done, infos = self.env.step(decoded)
            episode_reward = infos.get("total_reward", None) if done else None

            if not evaluate:
                r_n = [r] * self.args.N
                if getattr(self.args, 'use_reward_norm', False):
                    r_n = self.reward_norm(r_n)
                elif getattr(self.args, 'use_reward_scaling', False):
                    r_n = self.reward_scaling(r_n)
                next_obs_vec = convert_state(next_state)
                self.replay_buffer.store_transition(
                    step,
                    obs_n,
                    next_obs_vec,
                    v_n_current,
                    flat_actions,
                    logprobs,
                    r_n,
                    [done] * self.args.N
                )

            obs_n = [convert_state(next_state)] * self.args.N
            if done:
                break

        if not evaluate:
            last_val = self.agent_n.get_value(obs_n[0])
            self.replay_buffer.store_last_value(step+1, last_val.tolist())

        # Log episode reward to wandb
        wandb.log({"episode_reward": episode_reward}, step=self.total_steps)
        return episode_reward, step+1

    def evaluate_policy(self):
        total = 0.0
        for _ in range(self.args.evaluate_times):
            r, _ = self.run_episode(evaluate=True)
            total += r
        avg = total / self.args.evaluate_times
        print(f"[Eval] steps={self.total_steps} reward={avg}")
        # Log evaluation reward
        wandb.log({"eval_reward": avg}, step=self.total_steps)
        # save only actor parameters
        torch.save(self.agent_n.actor.state_dict(), f"model_seed{self.seed}.pt")
