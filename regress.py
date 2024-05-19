import numpy as np
import pandas as pd
# import torch
import time
# from torch import nn
from MyDeepAR import *


class MyDeepARN(MyDeepAR):
    def __init__(self, **kwgs):
        super().__init__(**kwgs)
        self.min_loss = 10e6
        self.early_stop_sign = 0

    def construct_input_vector(self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None):
        # embedding x_cat
        embeddings = {name: self.embeddings[name](x_cat[..., i]) for i, name in enumerate(self.categoricals)}
        flat_embeddings = torch.cat([embeddings[name] for name in self.categoricals], dim=-1)

        # concat with x_cont
        input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]
        # shift target
        return input_vector

    def decode(self,
               input_vector: torch.Tensor,
               target_scale: torch.Tensor,
               hidden_state,
               n_samples: int = None, ) -> torch.Tensor:
        if n_samples is None:
            # run in train and validation
            output, _ = self.rnn(input_vector, hidden_state)  # LSTM decoder process
            output = self.distribution_projector(output)  # Liner projector process
            # every batch to scale  [target_scale[0], target_scale[1], loc, scale(softplus_function)]
            output = self.loss.rescale_parameters(parameters=output,
                                                  target_scale=target_scale,
                                                  encoder=self.output_transformer)
        else:
            # run in test and validation
            # for every batch，sample n_samples, get n_samples trace
            target_pos = self.target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, dim=0)  # [n_samples*batch, t, f]
            hidden_state = self.LSTMrepeat_interleave(hidden_state, n_samples)
            target_scale = target_scale.repeat_interleave(n_samples, 0)  # [6400,2]

            # define function to run at every decoding step
            def decode_one(idx, lagged_targets, hidden_state_one):
                x = input_vector[:, [idx]]  # 获得当前步的inputs
                x[:, 0, target_pos] = lagged_targets[-1]  # 使用预测norm的结果替换
                decoder_output, hidden_state_one = self.rnn(x, hidden_state_one)  # LSTM
                prediction = self.distribution_projector(decoder_output)  # gaussian 分布，还要log(1+exp(\sigma))
                prediction = prediction[:, 0]  # select first time step
                return prediction, hidden_state_one

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],  #
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),  # time step
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = output.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1)
        return output

    def training_step(self, batch, device):
        # encode
        x, y = batch
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        _, hidden_state = self.rnn(input_vector.to(device))
        # decode
        one_off_target = x["encoder_cont"][:, -1, self.target_positions.unsqueeze(-1)].reshape(-1, 1)
        input_vector = self.construct_input_vector(x["decoder_cat"],  # [64,24,18]
                                                   x["decoder_cont"],
                                                   one_off_target)

        y_hat = self.decode(input_vector=input_vector.to(device),
                            target_scale=x['target_scale'].to(device),  # [64, 2]
                            hidden_state=hidden_state,
                            n_samples=None)
        y_true, _ = y  # [64, 24] (y_hat_orspace - y[0]).abs().mean()
        loss = self.loss.loss(y_hat, y_true.to(self.device)).sum()
        return loss, y_true.numel()

    def mask_p(self, epoch):
        p = min(1, 0 + epoch * 1 / 7 / 38)
        return p

    def validation_step(self, batch, device):
        # encode
        x, y = batch
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        _, hidden_state = self.rnn(input_vector.to(device))
        # deocode
        one_off_target = x["encoder_cont"][:, -1, self.target_positions.unsqueeze(-1)].reshape(-1, 1)
        input_vector = self.construct_input_vector(x["decoder_cat"],
                                                   x["decoder_cont"],
                                                   one_off_target)

        y_hat = self.decode(input_vector=input_vector.to(device),  # [64, 24, 4]
                            target_scale=x['target_scale'].to(device),
                            hidden_state=hidden_state,
                            n_samples=None)
        y_true, _ = y  # [64, 24] (y_hat_orspace - y[0]).abs().mean()
        loss = self.loss.loss(y_hat, y_true.to(self.device)).sum()
        return loss, y_true.numel()

    def prediction(self, test_dl, n_samples=100):
        out = []
        x_list = []
        y_list = []
        y_pred_list = []
        decoder_y = []
        encoder_y = []
        decoder_cat = []
        decoder_time_index = []

        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dl):
                # encode
                x, y = batch
                input_vector = self.construct_input_vector(x["encoder_cat"].clone(), x["encoder_cont"].clone())
                _, hidden_state = self.rnn(input_vector.to(self.device))

                one_off_target = x["encoder_cont"][:, -1, self.target_positions.unsqueeze(-1)].reshape(-1, 1)
                input_vector = self.construct_input_vector(x["decoder_cat"].clone(),
                                                           x["decoder_cont"].clone(),
                                                           one_off_target)

                y_hat = self.decode(input_vector=input_vector,  # [64, 24, 100]
                                    target_scale=x['target_scale'].clone(),
                                    hidden_state=hidden_state,
                                    n_samples=n_samples)

                # y_hat_orspace = (y_hat[..., 2] * y_hat[..., 1] + y_hat[..., 0] - y[0]).abs()

                out.append(y_hat)  # 记录预测结果
                decoder_y.append(y[0])  # 记录真实的y
                encoder_y.append(x['encoder_target'])  # 记录encoder部分的值
                #                 x_list.append(x)  # 保存x
                #                 y_list.append(y)  # 保存y
                decoder_cat.append(x['decoder_cat'])
                decoder_time_index.append(x['decoder_time_idx'])

            out = torch.cat(out, dim=0)
            decoder_y = torch.cat(decoder_y, dim=0)
            encoder_y = torch.cat(encoder_y, dim=0)
            decoder_cat = torch.cat(decoder_cat, dim=0)
            decoder_time_index = torch.cat(decoder_time_index, dim=0)
        return out, decoder_y, encoder_y, (decoder_cat, decoder_time_index)  # , x_list, y_list

    # 调用预测算法 regress中
    # [w_k1,p_k1] [w_k]  ->  [p_k]
    def get_next_day_power(self, w_k1: [], p_k1: [], w_k: [], train_dataset, start_str, end_str):
        # 上面的数据变为DataFrame 大小为48 * N
        load_all = [it[0] for it in p_k1] + [it[0] for it in p_k1]
        temp = [it[0] for it in w_k1] + [it[0] for it in w_k]
        time_index = pd.date_range(start=start_str, end=end_str, freq='h')
        load_data = pd.DataFrame({'time': time_index, 'load': load_all, 'temp': temp})
        load_data['year'] = load_data['time'].dt.year
        load_data['week'] = load_data['time'].dt.weekday
        load_data['month'] = load_data['time'].dt.month
        load_data['hour'] = load_data['time'].dt.hour
        load_data['Time_index'] = (load_data['time'].dt.day - 1) * 24 + load_data['time'].dt.hour  # 为了对数据依据各月份分别划分
        load_data = load_data.astype(dict(hour=str, week=str, month=str))

        run_dataset = TimeSeriesDataSet.from_dataset(train_dataset, load_data)
        batch_size = 128
        run_dl = run_dataset.to_dataloader(train=False,
                                           num_workers=0,
                                           batch_sampler=BatchSampler(SequentialSampler(run_dataset),
                                                                      batch_size=batch_size,
                                                                      drop_last=False)
                                           )
        # 预测
        out, _, _, (_, _) = self.prediction(run_dl, n_samples=100)
        self.loss.quantiles = [0.1, 0.5, 0.99]
        y_quantiles = self.to_quantiles(out, use_metric=False)
        p1 = [it.item() for it in y_quantiles[:, :, 0].reshape(-1)]
        p50 = [it.item() for it in y_quantiles[:, :, 1].reshape(-1)]
        p99 = [it.item() for it in y_quantiles[:, :, 2].reshape(-1)]
        return p1, p50, p99


def get_load_data():
    # data = pd.read_excel('data/泗洪各行业分时电量.xlsx')
    # cats = ['工业']
    # cats_len = len(cats)
    # # 分别获得各个行业数据
    # index = data.columns[2:]
    # data_cats = [data[data['行业大类一级分类'] == cat].reset_index() for cat in cats]
    # # 对各个行业数据依次处理
    # for data_cat in data_cats:
    #     # 获得第i时刻的数据
    #     for idx in index:
    #         # 提取平均值并替换掉str和异常值
    #         # 提取useful_data
    #         useful_data = []
    #         for i in data_cat[idx]:
    #             if type(i) == str:
    #                 continue
    #             if float(i):
    #                 useful_data.append(i)
    #
    #         # 异常值识别并求取mean
    #         Q1 = np.quantile(useful_data, 0.25)
    #         Q3 = np.quantile(useful_data, 0.75)
    #         IQR = Q3 - Q1
    #         k = 0.5
    #         point_max = Q3 + k * IQR
    #         point_min = max(Q1 - k * IQR, 0)
    #         useful_data = np.array(useful_data)
    #         mean = useful_data[(useful_data < point_max) & (useful_data > point_min)].mean()
    #         # 替换掉data_cat 中 idx处的值
    #         replace_data = []
    #         for i in data_cat[idx]:
    #             if type(i) == str:
    #                 replace_data.append(mean)
    #             elif (float(i) > point_min) & (float(i) < point_max):
    #                 replace_data.append(i)
    #             else:
    #                 replace_data.append(mean)
    #
    #         data_cat[idx] = replace_data
    # # 构造时间序列数据
    # load_all = np.array(data_cats[0].iloc[:, 3:-1]).reshape(-1)
    # for i in range(1, len(cats)):
    #     load_all += np.array(data_cats[i].iloc[:, 3:-1]).reshape(-1)
    #
    # time_index = pd.date_range(start='20220524 08:00:00', end='20230514 07:00:00', freq='h')
    # load_data = pd.DataFrame({
    #     'time': time_index,
    #     'load': load_all
    # })
    # load_data['year'] = load_data['time'].dt.year
    # load_data['week'] = load_data['time'].dt.weekday
    # load_data['month'] = load_data['time'].dt.month
    # load_data['hour'] = load_data['time'].dt.hour
    # load_data['Time_index'] = (load_data['time'].dt.day - 1) * 24 + load_data['time'].dt.hour  # 为了对数据依据各月份分别划分
    # load_data.index = time_index
    #
    # load_data = load_data.loc['2022-05-24 08:00:00':'2023-03-31 07:00:00']
    # weather_data = pd.read_csv('data/泗洪气象数据.csv')
    # load_data['temp'] = np.array(weather_data['temperature_2m']) - 273.15
    # load_data['load'] = load_data['load'] * 1.3 / 1000
    # load_data = load_data[16:-8]
    #
    # # 建模数据
    # load_data = load_data.loc['2022-06-1 00:00:00':'2023-03-30 23:00:00']
    # load_data.to_excel('my_data.xlsx', index=False)
    data = pd.read_excel('data/indust_model_data.xlsx')
    data = data.astype(dict(hour=str, week=str, month=str))
    time_varying_known_categoricals = ['hour', 'week', 'month']
    time_varying_known_reals = ['temp']
    encoder_length = 24
    decoder_length = 24

    train_dataset = TimeSeriesDataSet(
        data=data[lambda x: x.Time_index < 520],
        time_idx="Time_index",
        target="load",
        group_ids=["month"],
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["load"],
        max_encoder_length=encoder_length,
        max_prediction_length=decoder_length,
        allow_missing_timesteps=True,
        target_normalizer=EncoderNormalizer(transformation='relu', method='standard')
    )
    return train_dataset


def get_DeepAR(train_dataset):
    device = torch.device('cpu')
    embeddings = torch.load("model/embeddings.pth")
    net = MyDeepARN.from_dataset(train_dataset, hidden_size=64, layers=2, dropout=0.1, device=device)
    net.embeddings = embeddings
    net.load_state_dict(torch.load("model/DeepAR.pth"))  # get_next_day_power 主要接口
    return net


# 评价测试集
def val_epoch_loss(data_iter, net, device=None):
    if device is None:
        device = list(net.parameters())[0].device

    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in data_iter:
            net.eval()
            l, ynumel = net.validation_step(batch, device)
            loss_sum += l.cpu().item()
            n += ynumel

    return loss_sum / n


def call_save_model(net, val_loss):
    if val_loss < net.min_loss:
        net.min_loss = val_loss
        torch.save(net.state_dict(), "model/DeepAR.pth")
        print(f'call:{val_loss}')


def call_early_stop(net, val_loss):
    if val_loss < net.min_loss:
        net.min_loss = val_loss
        torch.save(net.state_dict(), "model/DeepAR.pth")
        print(f'call:{val_loss}')
        net.early_stop_sign = 0
    else:
        net.early_stop_sign += 1
    #       torch.save(net.state_dict(), "model/DeepAR.pth")

    if net.early_stop_sign >= 5:
        return True
    else:
        return False


def train_net(net, train_iter, test_iter, optimizer, device, epochs):
    global seed
    net = net.to(device)
    test_loss_list = []
    train_loss_list = []

    for epoch in range(1, epochs + 1):
        # setup_seed(seed+epoch)
        train_l_sum, train_mae_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch in train_iter:
            net.train()
            l, ynumel = net.training_step(batch, device)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                train_l_sum += l.cpu().item()
                n += ynumel
        test_loss = val_epoch_loss(test_iter, net, device)

        test_loss_list.append(test_loss)
        train_loss_list.append(train_l_sum / n)

        print('epoch %d, train loss %.4f, test loss %.3f, time %.2f sec'
              % (epoch, train_l_sum / n, test_loss, time.time() - start))

        # 保存模型
        # call_save_model(net,  test_mae)
        early_stop = call_early_stop(net, test_loss)
        if early_stop:
            break


if __name__ == "__main__":
    # 读取数据处理并获得训练模型
    # load_data = get_load_data()
    # data = load_data.loc['2022-06-1 00:00:00':]
    # data = data.astype(dict(hour=str, week=str, month=str))
    # time_varying_known_categoricals = ['hour', 'week', 'month']
    # time_varying_known_reals = ['temp']
    # encoder_length = 24
    # decoder_length = 24
    #
    # train_dataset = TimeSeriesDataSet(
    #     data=data[lambda x: x.Time_index < 520],
    #     time_idx="Time_index",
    #     target="load",
    #     group_ids=["month"],
    #     static_categoricals=[],
    #     static_reals=[],
    #     time_varying_known_categoricals=time_varying_known_categoricals,
    #     time_varying_known_reals=time_varying_known_reals,
    #     time_varying_unknown_categoricals=[],
    #     time_varying_unknown_reals=["load"],
    #     max_encoder_length=encoder_length,
    #     max_prediction_length=decoder_length,
    #     allow_missing_timesteps=True,
    #     target_normalizer=EncoderNormalizer(transformation='relu', method='standard')
    # )
    # val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, data[lambda x: x.Time_index < 620],
    #                                              min_prediction_idx=520)
    # test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, data, min_prediction_idx=620)

    # train_dataset = get_load_data()
    # run_data = get_power_weather()
    # run_dataset = TimeSeriesDataSet.from_dataset(train_dataset, run_data)

    # seed=20
    # setup_seed(seed)
    # batch_size = 128
    # train_dl = train_dataset.to_dataloader(train=True,
    #                                        num_workers=0,
    #                                        batch_sampler=BatchSampler(RandomSampler(train_dataset),
    #                                                                   batch_size=batch_size,
    #                                                                   drop_last=True),
    #                                        )

    # val_dl = val_dataset.to_dataloader(train=True,
    #                                    num_workers=0,
    #                                    batch_sampler=BatchSampler(SequentialSampler(val_dataset),
    #                                                               batch_size=batch_size,
    #                                                               drop_last=False)
    #                                    )
    #
    # test_dl = test_dataset.to_dataloader(train=False,
    #                                      num_workers=0,
    #                                      batch_sampler=BatchSampler(SequentialSampler(test_dataset),
    #                                                                 batch_size=batch_size,
    #                                                                 drop_last=False)
    #                                      )

    # run_dl = run_dataset.to_dataloader(train=False,
    #                                      num_workers=0,
    #                                      batch_sampler=BatchSampler(SequentialSampler(run_dataset),
    #                                                                 batch_size=batch_size,
    #                                                                 drop_last=False)
    #                                      )
    # device = torch.device('cpu')
    # print('training on', device)
    # 保存模型的embeddings，接下来所有模型使用相同的embedding
    # net = MyDeepARN.from_dataset(train_dataset, hidden_size=64, layers=2, dropout=0.1, device=device)
    # torch.save(net.embeddings, "model/embeddings.pth")
    # embeddings = torch.load("model/embeddings.pth")
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # epochs = 50
    # train_net(net, train_dl, val_dl, optimizer, device, epochs)
    # 加载最新模型
    # 加载模型
    # device = torch.device('cpu')
    # embeddings = torch.load("model/embeddings.pth")
    # net = MyDeepARN.from_dataset(train_dataset, hidden_size=64, layers=2, dropout=0.1, device=device)
    # net.embeddings = embeddings
    # net.load_state_dict(torch.load("model/DeepAR.pth"))

    # r = 0.9
    # # 预测
    # out, y_true, encoder_y, (decoder_cat, decoder_time_index) = net.prediction(run_dl, n_samples=100)
    # encoder_target = encoder_y
    # decoder_target = y_true
    # # y_hats = net.to_prediction(out, use_metric=False)  # mean
    # net.loss.quantiles = [0.1, 0.5, 0.99]
    # y_quantiles = net.to_quantiles(out, use_metric=False)
    print('test')
