from torch import nn
import torch.nn.functional as F

import pfrl
from pfrl.policies import SoftmaxCategoricalHead

def make_A2C_model(obs_size, n_hidden_size, n_actions):
    model = nn.Sequential(
        nn.Linear(obs_size, n_hidden_size),
        nn.ReLU(),
        nn.Linear(n_hidden_size, n_hidden_size),
        nn.ReLU(),
        nn.Linear(n_hidden_size, n_hidden_size),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(n_hidden_size, n_actions),
                SoftmaxCategoricalHead(),
            ),
            nn.Linear(n_hidden_size, 1),
        ),
    )
    return model
