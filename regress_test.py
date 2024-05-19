import torch
from torch import nn
import numpy.random as nprandom


class RNNModel(nn.Module):
    def __init__(self, enc_feature_size, dec_feature_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.enc_feature_size = enc_feature_size
        self.dec_feature_size = dec_feature_size
        self.num_hiddens = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = nn.LSTM(input_size=enc_feature_size, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout)
        self.decoder = nn.LSTM(input_size=dec_feature_size, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.num_hiddens, 16),
            nn.Linear(16, 1))

        self.val_loss_list = []  # 存储验证集每个epoch的损失
        self.train_loss_list = []  # 存储训练集每个epoch的损失
        self.min_loss = 10e6
        self.early_stop_sign = 0

    def encode(self, x_enc, state0):
        _, state = self.encoder(x_enc, state0)
        return state

    def decode(self, x_dec, state):
        H, state = self.decoder(x_dec, state)
        y_hat = self.fc(H.reshape(-1, self.num_hiddens))
        return y_hat, state

    def forward(self, x_enc, x_dec):
        state = self.encode(x_enc)
        y_hat, _ = self.decode(x_dec, state)
        return y_hat

    def init_state(self, batch_size, device):
        return (torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device))


def get_next_day_power(w_k1, p_k1, w_k, net, device):
    # 建模时负荷单位设置为 MW
    # data 依次为 power, month, day, weekday, hour, temp, hum, wind_speed
    x_enc = [[itemp[0], itemw[0].month, itemw[0].day, itemw[0].weekday(), itemw[1],
              itemw[2], itemw[3], itemw[4]] for itemp, itemw in zip(p_k1, w_k1)]
    # data 依次为 month, day, weekday, hour, temp, hum, wind_speed
    x_dec = [[itemw[0].month, itemw[0].day, itemw[0].weekday(), itemw[1],
              itemw[2], itemw[3], itemw[4]] for itemw in w_k]

    x_enc = torch.tensor(x_enc)  # 1*24*8
    x_dec = torch.tensor(x_dec)  # 1*24*7
    # 归一化
    max_norm = torch.tensor([4000, 12, 31, 6, 23, 40, 100, 7])
    min_norm = torch.tensor([800, 1, 1, 0, 0, -10, 0, 0])
    task_max = 4000
    task_min = 800
    x_enc = ((x_enc - min_norm) / (max_norm - min_norm)).unsqueeze(0)  # 1*24*8
    x_dec = ((x_dec - min_norm[1:]) / (max_norm[1:] - min_norm[1:])).unsqueeze(0)   # 1*24*7
    x_enc.transpose_(0, 1)
    x_dec.transpose_(0, 1)
    x_enc = x_enc.to(device=device, dtype=torch.float32)
    x_dec = x_dec.to(device=device, dtype=torch.float32)
    state0 = net.init_state(batch_size=x_enc.shape[1], device=device)
    with torch.no_grad():
        net.eval()
        state = net.encode(x_enc, state0)
        y_hat, _ = net.decode(x_dec, state)
    y_hat = y_hat.reshape(-1)*(task_max - task_min) + task_min
    return y_hat.tolist(), y_hat.tolist(), y_hat.tolist()


def get_next_day_power_k1(w_k1, p_k1, w_k):
    p_k = [item[0] for item in p_k1]
    p_k = [item + nprandom.normal(loc=0, scale=0.08 * abs(item)) for item in p_k]
    return p_k, p_k, p_k
