import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
from layers.PQNLayer import PQN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
       
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        if self.pre_length in (96,192):
            self.hidden_size = 256
        else:
            self.hidden_size = 512
        self.res_fc = nn.Sequential(
            nn.Linear(self.seq_length, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.pre_length, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.PQN = PQN(self.seq_length,self.pre_length)

    
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, L, C = x.shape
        
        x = x.permute(0,2,1) # [B C L]
       
        bias = self.res_fc(x)

        x = self.PQN(x)

        x = x + bias

        x = self.fc(x).permute(0, 2, 1)
        # x = x.permute(0,2,1)
        return x
    
if __name__ == '__main__':
    class Configs(object):
        channel_independence=0
        exp_setting = 0
        ab = 0
        modes = 32
        mode_select = 'random'
        version = 'Fourier'
        # version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs).to('cuda')

    print('parameter number is {} M'.format(sum(p.numel() for p in model.parameters())/1e6))
    x = torch.randn([3, configs.seq_len, 7]).to('cuda')
    y = model(x)
    print(y.shape)
    # enc_mark = torch.randn([3, configs.seq_len, 4])

    # dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    # dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    # out = model.forward(enc, enc_mark, dec, dec_mark)
    # print(out.size)