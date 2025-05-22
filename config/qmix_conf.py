class QMIX_Args:
    def __init__(self):
        self.N = 5
        self.action_dim = 15
        self.obs_dim = None
        self.state_dim = None
        
        self.max_train_steps = 1e6
        self.episode_limit = 100
        self.evaluate_freq = 5000
        self.evaluate_times = 1
        self.save_freq = 1e5
        
        self.algorithm = "QMIX"
        self.epsilon = 1.0
        self.epsilon_decay_steps = 50000
        self.epsilon_min = 0.05
        self.buffer_size = 5000
        self.batch_size = 32
        self.lr = 5e-4
        self.gamma = 0.99
        self.qmix_hidden_dim = 32
        self.hyper_hidden_dim = 64
        self.hyper_layers_num = 1
        self.rnn_hidden_dim = 64
        self.mlp_hidden_dim = 64
        self.use_rnn = True
        self.use_orthogonal_init = True
        self.use_grad_clip = True
        self.use_lr_decay = False
        self.use_RMS = False
        self.add_last_action = True
        self.add_agent_id = True
        self.use_double_q = True
        self.use_reward_norm = False
        self.use_hard_update = True
        self.target_update_freq = 200
        self.tau = 0.005
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps
        