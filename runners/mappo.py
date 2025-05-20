import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from normalization import Normalization, RewardScaling
from algorithms.replay_buffer import ReplayBufferPPO
from algorithms import MAPPO
from .env_wrapper import Env, reward_shaping

class Runner_MAPPO:
    def __init__(
        self,
        map_file: str,
        max_train_steps: int = int(3e6),
        episode_limit: int = 100,
        batch_size: int = 32,
        mini_batch_size: int = 8,
        rnn_hidden_dim: int = 64,
        mlp_hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        lamda: float = 0.95,
        epsilon: float = 0.2,
        K_epochs: int = 15,
        entropy_coef: float = 0.01,
        use_adv_norm: bool = True,
        use_reward_norm: bool = False,
        use_reward_scaling: bool = False,
        use_lr_decay: bool = True,
        use_grad_clip: bool = True,
        set_adam_eps: bool = True,
        use_rnn: bool = False,
        add_agent_id: bool = False,
        use_value_clip: bool = False,
        reward_shaping_fn=reward_shaping,
        seed: int = 2025
    ):
        # devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        # create gym env
        self.env = Env(map_file, max_time_steps=episode_limit)
        # dims
        obs_example, _ = self.env.reset()
        obs_dim = obs_example.shape[0]
        action_branches = self.env.action_space.nvec
        joint_action_dim = int(np.prod(action_branches))
        # params
        self.max_train_steps = max_train_steps
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.reward_shaping = reward_shaping_fn
        # agent
        self.agent = MAPPO(
            N=1,
            action_dim=joint_action_dim,
            obs_dim=obs_dim,
            state_dim=obs_dim,
            episode_limit=episode_limit,
            rnn_hidden_dim=rnn_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            max_train_steps=max_train_steps,
            lr=lr,
            gamma=gamma,
            lamda=lamda,
            epsilon=epsilon,
            K_epochs=K_epochs,
            entropy_coef=entropy_coef,
            set_adam_eps=set_adam_eps,
            use_grad_clip=use_grad_clip,
            use_lr_decay=use_lr_decay,
            use_adv_norm=use_adv_norm,
            use_rnn=use_rnn,
            add_agent_id=add_agent_id,
            use_value_clip=use_value_clip
        )
        # move models to device
        self.agent.actor.to(self.device)
        self.agent.critic.to(self.device)
        # buffer
        self.buffer = ReplayBufferPPO(
            n_agents=1,
            obs_dim=obs_dim,
            state_dim=obs_dim,
            episode_limit=episode_limit,
            batch_size=batch_size
        )
        # tb writer
        self.writer = SummaryWriter(log_dir=f"runs/CustomMAPPO/{seed}")
        self.total_steps = 0

    def encode_action(self, a_idx: int):
        dims = self.env.action_space.nvec
        coords = []
        idx = a_idx
        for base in reversed(dims):
            coords.append(idx % base)
            idx //= base
        return np.array(coords[::-1], dtype=int)

    def run(self):
        eval_count = 0
        while self.total_steps < self.max_train_steps:
            if self.total_steps >= eval_count * self.episode_limit:
                self.evaluate_policy()
                eval_count += 1
            self.collect_episode(train=True)
        self.evaluate_policy()

    def collect_episode(self, train=False):
        obs, _ = self.env.reset()
        if train and self.agent.use_rnn:
            self.agent.actor.rnn_hidden = None
            self.agent.critic.rnn_hidden = None
        for t in range(self.episode_limit):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            probs = self.agent.actor(obs_tensor).detach().cpu().numpy().flatten()
            if not train:
                a_idx = np.argmax(probs)
            else:
                a_idx = np.random.choice(len(probs), p=probs/np.sum(probs))
            coords = self.encode_action(a_idx)
            next_obs, r, done, _ ,info = self.env.step(coords)
            episode_reward = info.get("total_reward", None) if done else None
            r_shaped = self.reward_shaping(r, self.env, obs, coords)
            if train:
                v = self.agent.get_value(obs)
                self.buffer.store_transition(
                    episode_step=t, obs_n=obs, s=obs,
                    v_n=v, a_n=[a_idx], a_logprob_n=[0.],
                    r_n=[r_shaped], done_n=[done]
                )
            obs = next_obs
            self.total_steps += 1
            if done:
                break
        if train:
            v = self.agent.get_value(obs)
            self.buffer.store_last_value(min(t+1, self.episode_limit), v)
            if self.buffer.episode_num >= self.buffer.batch_size:
                # move batch to device inside train
                self.agent.train(self.buffer, self.total_steps)
                self.buffer.reset_buffer()
            return episode_reward
        else:
            return episode_reward

    def evaluate_policy(self):
        total_r = 0
        for _ in range(1):
            total_r += self.collect_episode(train=False)
        avg_r = total_r / 1
        print(f"Step {self.total_steps}, Eval reward {avg_r:.2f}")
        self.writer.add_scalar("eval_reward", avg_r, self.total_steps)

# if __name__ == '__main__':
#     runner = Runner_MAPPO_Custom('map.txt')
#     runner.run()
