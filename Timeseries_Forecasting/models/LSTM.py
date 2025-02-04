import torch
import torch.nn as nn
import sys
sys.path.append('..')

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.feature_size = configs.enc_in  # 输入特征数 (channels)
        self.seq_length = configs.seq_len
        self.hidden_size = 256 if self.pred_len in (96, 192) else 512
        self.num_layers = 1  # LSTM 层数

        # 定义 LSTM
        self.lstm = nn.LSTM(
            input_size=self.seq_length,
            hidden_size=self.pred_len,
            num_layers=self.num_layers,
            batch_first=True
        )
       
        self.fc = nn.Sequential(
            nn.Linear(self.pred_len, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def forward(self, x):
        # x: [Batch, Sequence length, Channels]
        B, L, C = x.shape

        x = x.permute(0,2,1) # [B C L]

        # LSTM 期望输入: [Batch, Sequence length, Input size]
        out, (hn, cn) = self.lstm(x)

        
        # print(out.shape)
        # 提取最后一个时间步的输出，或者直接用全连接层对所有时间步进行预测
        # out = self.fc(out).permute(0, 2, 1)
        out = out.permute(0, 2, 1)
        # print(out.shape)
        return out


if __name__ == '__main__':
    class Configs(object):
        channel_independence = 0
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
        pred_len = 360
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

    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    x = torch.randn([3, configs.seq_len, 7]).to('cuda')
    y = model(x)
    print(y.shape)