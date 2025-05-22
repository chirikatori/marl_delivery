class MAPPO_Args:
    def __init__(self):
        self.N = 5
        self.action_dim = 15
        self.obs_dim = None
        self.state_dim = None

        self.max_train_steps = 3e6
        self.episode_limit = 100
        self.evaluate_freq = 5000
        self.evaluate_times = 1
        
        self.batch_size = 32
        self.mini_batch_size = 8
        self.rnn_hidden_dim = 64
        self.mlp_hidden_dim = 64
        self.lr = 5e-4
        self.gamma = 0.99
        self.lamb = 0.95
        self.epsilon = 0.2
        self.K_epochs = 15
        
        self.use_adv_norm = True
        self.use_reward_norm = True
        self.use_reward_scaling = False
        self.entropy_coef = 0.01
        self.use_lr_decay = True
        self.use_grad_clip = True
        self.use_orthogonal_init = True
        self.set_adam_eps = True
        self.use_relu = False
        self.use_rnn = False
        self.add_agent_id = False
        self.use_value_clip = False
        