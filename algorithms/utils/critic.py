import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)

class Critic_RNN(nn.Module):
    def __init__(self,
                 critic_input_dim: int,
                 rnn_hidden_dim: int,
                 use_relu: bool = True,
                 use_orthogonal_init: bool = False):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][int(use_relu)]

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value
    

class Critic_MLP(nn.Module):
    def __init__(self,
                 critic_input_dim: int,
                 mlp_hidden_dim: int,
                 use_relu: bool = True,
                 use_orthogonal_init: bool = False):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc3 = nn.Linear(mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][int(use_relu)]

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value