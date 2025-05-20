import os

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class RNN(nn.Module):
    def __init__(self,
                 input_shape: int,
                 n_actions: int,
                 rnn_hidden_dim: int = 128):
        super(RNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, n_actions)
    
    def forward(self, obs, hidden_state):
        x = torch.relu(self.fc1(obs))
        h_in = hidden_state.view(-1, self.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        q_values = self.fc2(h_out)
        return q_values, h_out


class QMixNet(nn.Module):
    