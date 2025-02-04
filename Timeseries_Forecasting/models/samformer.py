import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.utils.attention import scaled_dot_product_attention
from models.utils.dataset import LabeledDataset
from models.utils.revin import RevIN
from models.utils.sam import SAM
from layers.PQNLayer import PQN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.use_revin = configs.use_revin if hasattr(configs, 'use_revin') else True
        self.num_channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_horizon = configs.pred_len
        self.hid_dim = configs.hid_dim if hasattr(configs, 'hid_dim') else 16

        self.revin = RevIN(num_features=self.num_channels) if self.use_revin else None
        self.compute_keys = nn.Linear(self.seq_len, self.hid_dim)
        self.compute_queries = nn.Linear(self.seq_len, self.hid_dim)
        self.compute_values = nn.Linear(self.seq_len, self.seq_len)
        self.linear_forecaster = nn.Linear(self.seq_len, self.pred_horizon)
        
        self.PQN = PQN(self.seq_len, self.seq_len)
        self.fc = nn.Linear(self.seq_len, self.seq_len)
        self.res_fc = nn.Sequential(
            nn.Linear(self.seq_len, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.seq_len)
        )

    def forward(self, x):
        # x: [Batch, Channels, Sequence Length]
        if self.use_revin:
            x = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2)

        # Channel-Wise Attention
        queries = self.compute_queries(x)
        keys = self.compute_keys(x)
        values = self.compute_values(x)
        if hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values)
        # att_score = self.PQN(x)
        # x = self.fc(x)
        x = x + att_score
        x = self.linear_forecaster(x)

        if self.use_revin:
            out = self.revin(x.transpose(1, 2), mode='denorm')
        
        return out
        

