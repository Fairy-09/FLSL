# -*- coding: UTF-8 -*-

import pickle
import re
import socket
import time
import torch
from matplotlib.animation import FuncAnimation
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from support_SSA import SSA 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn import tree
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error as MSE1
from sklearn.metrics import mean_absolute_error as MAE1
from sklearn import metrics
import struct

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=4, output_size=1, num_layer=1,bidirectional=True):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,bidirectional=bidirectional)
        self.layer2 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = torch.relu(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)  
        return x[:,-1,:]


look_back = 10
EPOCH = 16    ##
head = [None for i in range(look_back)]
SIZE = 8500
original_mean = 0

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))
def add_noise(parameters):
    parameters = parameters.to(device)
    noise = torch.randn(parameters.shape, device=device).normal_(0, 0.01)
    return parameters.add_(noise)
def MAPE1(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_dataset(dataset):

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back): 
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])

    data_X = np.array(dataX)
    data_Y = np.array(dataY)
    train_X = data_X[:7000]
    train_Y = data_Y[:7000]
    val_X = data_X[7000:8000]
    val_Y = data_Y[7000:8000]
    test_X = data_X[8000:10000]
    test_Y = data_Y[8000:10000]

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    train_Y = train_Y.reshape(-1, 1, 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)
    val_Y = val_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    test_Y = test_Y.reshape(-1, 1, 1)        
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def __coeffs_to_list(coeffs, devicce):
    level_list = []
    for level in range(len(coeffs)):
        var_x, var_y, var_valX, val_Y,var_testX, test_Y = create_dataset(coeffs[level])
        level_part = [torch.from_numpy(var).float().to(device) for var in [var_x, var_y, var_valX, val_Y, var_testX, test_Y]]
        level_list.append(level_part)
    return level_list


def __preprocessing_SSA(data, device):
	result_ssa = SSA(data.reshape(-1, ), 4)
	result_ssa = result_ssa.tolist()
	for i in range(len(result_ssa)):
		result_ssa[i] = np.array(result_ssa[i]).reshape(-1, 1)
	result_ssa = __coeffs_to_list(result_ssa,device)           
    
	return result_ssa


def fl_data_load(path, device):
    dataframe = read_csv(path, engine='python')

    dataset = dataframe[0:10000].values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')
    np.random.seed(7)
    

    dataset = scaler.fit_transform(dataset)

    original_mean = np.mean(dataset)

    data_SSA = __preprocessing_SSA(dataset, device)

    return data_SSA, original_mean

##################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

path = r"../../Dataset/house4_5min_KWh.csv"
data_SSA, original_mean = fl_data_load(path, device)

time_steps, input_dim, output_dim = 10, 1, 1
test_size = 1000


model = LSTM(1, 4, 1, 2)
model.to(device) 
print(model)
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


losses = list()
steps = list()

################################# #-# ######################################
for epoch in range(1, EPOCH + 1):
    log("\033[1;31;40m第\033[1;31;40m%s\033[1;31;40m轮开始训练!\033[1;31;40m" % str(epoch))

    for subSignal_index in range(len(data_SSA)):
        print(str(subSignal_index+1) + '_level SSA data Start.')
#########

        for t in range(10):
            loss_t = list()   
            out = model(data_SSA[subSignal_index][0])
            loss = loss_fun(out, data_SSA[subSignal_index][1].squeeze(-1))
            print(loss)
            loss_t.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(sum(loss_t)/len(loss_t))
        steps.append(epoch)
        print(str(subSignal_index + 1) + '_level SSA data Complete!!\n')

    log("建立连接并上传......")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    host = '127.0.0.1'
    port = 7002

    s.connect((host, port))

    data = {}
    data['num'] = epoch
    data['model'] = model.state_dict()

    torch.save(model.state_dict(), 'model.pth')

    keys = model.state_dict().keys()
    data = pickle.dumps(data)

    data_length = len(data)
    s.sendall(data_length.to_bytes(4, byteorder='big'))

    s.sendall(data)

    log("等待接收......")
    try:
        s.settimeout(1000)  
        data_length_bytes = s.recv(4)
        if not data_length_bytes:
            raise ValueError("未接收到数据长度信息")
        data_length = int.from_bytes(data_length_bytes, 'big')

        received_data = b''
        while len(received_data) < data_length:
            packet = s.recv(data_length - len(received_data))
            if not packet:
                raise ConnectionError("连接中断")
            received_data += packet

        data = pickle.loads(received_data)
        print(data['num'], epoch)
        if data['num'] == epoch:
            global_state_dict = data['model']
        else:
            global_state_dict = model.state_dict()
    except Exception as e:
        print(e)
        # s.sendto(data, (host, port))
        log("没有在规定时间收到正确的包， 利用本地参数更新")
        global_state_dict = model.state_dict()

    model.load_state_dict(global_state_dict)
    s.close()
log("训练完毕，关闭连接")
s.close()

plt.plot(steps, losses, "o-")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title('Training and Validation Loss')
plt.legend()
plt.draw()
plt.pause(0.1)
plt.savefig('loss_curve4.png')
plt.close()

################################# #-# ######################################

predict_list = []
test_y_list = []
for subSignal_index in range(len(data_SSA)):
    print('No.' + str( subSignal_index+1) + 'Prediction Start.')
    with torch.no_grad():
        model.eval()

        predictions_pytorch_forecast = model(data_SSA[subSignal_index][4]).cpu()
        predict = predictions_pytorch_forecast
        predict_list.append(predict)
        test_y_list.append(data_SSA[subSignal_index][5].cpu())
    print('No.' + str( subSignal_index+1) + 'Prediction Complete.')


predict_array = np.stack([tensor.numpy() for tensor in predict_list])
summed_predictions = np.sum(predict_array, axis=0)
predict = summed_predictions.squeeze()
predict = predict + original_mean

test_y_array = np.stack([tensor.numpy() for tensor in test_y_list])
summed_test_y = np.sum(test_y_array, axis=0)
test_y = summed_test_y.squeeze()
test_y = test_y + original_mean

pred_testY_origin = scaler.inverse_transform(predict.reshape(-1, 1))    
test_Y_origin = scaler.inverse_transform(test_y.reshape(-1, 1))    
###############################


Dict_Prediction_data = {}
data_test = pd.DataFrame(test_Y_origin)
data_test.to_csv(r"./myresult/test4.csv", index=False, header=False)
data_pre = pd.DataFrame(pred_testY_origin)
data_pre.to_csv(r"./myresult/pre4.csv", index=False, header=False)

plt.figure()
plt.plot(test_Y_origin, label='Actual', color='blue')
plt.plot(pred_testY_origin, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Value')
plt.savefig('./myresult/res4.png')

MAPE = MAPE1(test_Y_origin,pred_testY_origin)
MSE = MSE1(test_Y_origin,pred_testY_origin)
RMSE = np.sqrt(MSE1(test_Y_origin,pred_testY_origin))
MAE = MAE1(test_Y_origin,pred_testY_origin)
R2 = metrics.r2_score(test_Y_origin, pred_testY_origin)

print("MAPE：{:.6f}".format(MAPE))
print("MAE：{:.6f}".format(MAE))
print("MSE：{:.6f}".format(MSE))
print("RMSE：{:.6f}".format(RMSE))
print("R2：{:.6f}".format(R2))







