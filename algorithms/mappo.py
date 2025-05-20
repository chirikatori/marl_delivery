from .utils import Actor_MLP, Actor_RNN, Critic_MLP, Critic_RNN

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler


class MAPPO:
    def __init__(
        self,
        N: int,
        action_dim: int,
        obs_dim: int,
        state_dim: int,
        episode_limit: int,
        rnn_hidden_dim: int,
        mlp_hidden_dim: int,
        batch_size: int,
        mini_batch_size: int,
        max_train_steps: int,
        lr: float,
        gamma: float,
        lamda: float,
        epsilon: float,
        K_epochs: int,
        entropy_coef: float,
        set_adam_eps: bool = False,
        use_grad_clip: bool = False,
        use_lr_decay: bool = False,
        use_adv_norm: bool = False,
        use_rnn: bool = True,
        add_agent_id: bool = False,
        use_value_clip: bool = False
    ):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MAPPO] Using device: {self.device}")

        # config
        self.N = N
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.episode_limit = episode_limit
        self.rnn_hidden_dim = rnn_hidden_dim
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.max_train_steps = max_train_steps
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.set_adam_eps = set_adam_eps
        self.use_grad_clip = use_grad_clip
        self.use_lr_decay = use_lr_decay
        self.use_adv_norm = use_adv_norm
        self.use_rnn = use_rnn
        self.add_agent_id = add_agent_id
        self.use_value_clip = use_value_clip

        # input dims
        self.actor_input_dim = obs_dim + (N if add_agent_id else 0)
        self.critic_input_dim = state_dim + (N if add_agent_id else 0)

        # networks
        if use_rnn:
            self.actor = Actor_RNN(self.actor_input_dim, rnn_hidden_dim, action_dim, use_relu=use_adv_norm, use_orthogonal_init=set_adam_eps)
            self.critic = Critic_RNN(self.critic_input_dim, rnn_hidden_dim, use_relu=use_adv_norm, use_orthogonal_init=set_adam_eps)
        else:
            self.actor = Actor_MLP(self.actor_input_dim, mlp_hidden_dim, action_dim, use_relu=use_adv_norm, use_orthogonal_init=set_adam_eps)
            self.critic = Critic_MLP(self.critic_input_dim, mlp_hidden_dim, use_relu=use_adv_norm, use_orthogonal_init=set_adam_eps)

        # to device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # optimizer
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=lr, eps=1e-5 if set_adam_eps else 1e-8)

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            obs_n = torch.tensor(obs_n, dtype=torch.float32, device=self.device)
            actor_inputs = [obs_n]
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N, device=self.device))
            inp = torch.cat(actor_inputs, dim=-1)
            prob = self.actor(inp)
            if evaluate:
                a = prob.argmax(dim=-1)
                return a.cpu().numpy(), None
            dist = Categorical(prob)
            a = dist.sample()
            return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()

    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.N,1)
            inputs = [s]
            if self.add_agent_id:
                inputs.append(torch.eye(self.N, device=self.device))
            inp = torch.cat(inputs, dim=-1)
            v = self.critic(inp)
            return v.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        # fetch batch and move to device
        batch = replay_buffer.get_training_data()
        for k in batch:
            batch[k] = batch[k].to(self.device)

        # compute GAE and targets
        adv_list, gae = [], 0
        with torch.no_grad():
            deltas = batch['r_n'] + self.gamma*batch['v_n'][:,1:]* (1-batch['done_n']) - batch['v_n'][:,:-1]
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:,t] + self.gamma*self.lamda*(1-batch['done_n'][:,t])*gae
                adv_list.insert(0, gae)
            adv = torch.stack(adv_list, dim=1)
            v_target = adv + batch['v_n'][:,:-1]
            if self.use_adv_norm:
                adv = ((adv - adv.mean())/(adv.std()+1e-5)).clamp(-10,10)

        actor_inp, critic_inp = self.get_inputs(batch)
        # PPO updates
        for _ in range(self.K_epochs):
            for idx in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # forward
                if self.use_rnn:
                    self.actor.rnn_hidden = None; self.critic.rnn_hidden = None
                    probs_seq, vals_seq = [], []
                    for t in range(self.episode_limit):
                        p = self.actor(actor_inp[idx,t].reshape(-1,self.actor_input_dim))
                        probs_seq.append(p.reshape(-1,self.N,self.action_dim))
                        v = self.critic(critic_inp[idx,t].reshape(-1,self.critic_input_dim))
                        vals_seq.append(v.reshape(-1,self.N))
                    probs_now = torch.stack(probs_seq, dim=1)
                    vals_now = torch.stack(vals_seq, dim=1)
                else:
                    probs_now = self.actor(actor_inp[idx])
                    vals_now = self.critic(critic_inp[idx]).squeeze(-1)

                dist = Categorical(probs_now)
                ent = dist.entropy().mean()
                logp_now = dist.log_prob(batch['a_n'][idx])
                ratios = torch.exp(logp_now - batch['a_logprob_n'][idx].detach())
                s1 = ratios * adv[idx]; s2 = torch.clamp(ratios,1-self.epsilon,1+self.epsilon)*adv[idx]
                loss_actor = -torch.min(s1,s2).mean() - self.entropy_coef*ent
                if self.use_value_clip:
                    v_old = batch['v_n'][idx,:,:-1].detach()
                    v_clip = torch.clamp(vals_now - v_old, -self.epsilon, self.epsilon)+v_old
                    loss_critic = torch.max((v_clip-v_target[idx])**2,(vals_now-v_target[idx])**2).mean()
                else:
                    loss_critic = ((vals_now-v_target[idx])**2).mean()

                self.ac_optimizer.zero_grad()
                (loss_actor+loss_critic).backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters,10)
                self.ac_optimizer.step()

        # lr decay
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def get_inputs(self, batch):
        obs = batch['obs_n']; s = batch['s']
        act_in = [obs]
        crit_in = [s.unsqueeze(2).repeat(1,1,self.N,1)]
        if self.add_agent_id:
            aid = torch.eye(self.N, device=self.device).unsqueeze(0).unsqueeze(0)
            aid = aid.repeat(self.batch_size,self.episode_limit,1,1)
            act_in.append(aid); crit_in.append(aid)
        return torch.cat(act_in,-1), torch.cat(crit_in,-1)

    def lr_decay(self, total_steps):
        lr_now = self.lr*(1-total_steps/self.max_train_steps)
        for pg in self.ac_optimizer.param_groups: pg['lr']=lr_now

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), f"./model/MAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{total_steps//1000}k.pth")

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load(f"./model/MAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth", map_location=self.device))

