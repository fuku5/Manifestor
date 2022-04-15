
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.meta import Meta_model_MLP, Meta_Transformer, Guesser_Transformer, User_Transformer, User_MLP, User_Transformer_Multi
from agents.a2c import make_A2C_model


train_params = {
    'Guesser_Transformer':
    {
        'flatten': False,
        'pre_func': lambda x: x.transpose(1, 0),
        'calc_n_in': lambda sample_obs: sample_obs.shape[1],
        'model': Guesser_Transformer,
        'kargs': {
            'n_feature': 64,
            'n_out': 3, 'n_head': 2, 'n_layers': 2,
            'n_hidden': 1024, 'dropout': 0.5, 'encode': True
            }
    },
    'Transformer_state_seq':
    {
        'flatten': False,
        'pre_func': lambda x: x.transpose(1, 0),
        'calc_n_in': lambda sample_obs: sample_obs.shape[1],
        'model': Meta_Transformer,
        'kargs': {
            'n_feature': 32,
            'n_out': 3, 'n_head': 2, 'n_layers': 2,
            'n_hidden': 1024, 'dropout': 0.5, 'encode': True
            }
    },
    'Transformer_element_seq':
    {
        'flatten': False,
        'pre_func': lambda x: x.permute(2, 0, 1),
        'calc_n_in': lambda sample_obs: sample_obs.shape[0],
        'model': Meta_Transformer,
        'kargs': {
            'n_feature': 8,
            'n_out': 3, 'n_head': 2, 'n_layers': 2,
            'n_hidden': 1024, 'dropout': 0.5, 'encode': True
            }
    },
    'MLP':
    {
        'flatten': True,
        'pre_func': lambda x: x,
        'calc_n_in': lambda sample_obs: sample_obs.shape[0],
        'model': Meta_model_MLP,
        'kargs': {
            'n_out': 3, 'n_hidden': 256
            }
    }
}
