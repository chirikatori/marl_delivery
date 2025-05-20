import torch
import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self,
                 actor_input_dim: int,
                 rnn_hidden_dim: int,
                 action_dim: int,
                 use_relu: bool = True,
                 use_orthogonal_init: bool = False):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][int(use_relu)]

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Actor_MLP(nn.Module):
    def __init__(self,
                 actor_input_dim: int,
                 mlp_hidden_dim: int,
                 action_dim: int,
                 use_relu: bool = True,
                 use_orthogonal_init: bool = False):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc3 = nn.Linear(mlp_hidden_dim, action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][int(use_relu)]

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob
