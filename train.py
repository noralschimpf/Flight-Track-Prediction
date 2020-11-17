import numpy as np
import csv
import os
import torch
from custom_dataset import CustomDataset
from custom_dataset import ToTensor
import torch.nn as nn
from conv_lstm import CONV_LSTM


FP_root_dir = 'data/Dataset_test_data/FP'
FT_root_dir = 'data/Dataset_test_data/FT'
FP_data = CustomDataset(FP_root_dir, ToTensor())
FT_data = CustomDataset(FT_root_dir, ToTensor())

# train_model
model = CONV_LSTM() #customized Convolution and LSTM model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
# set training epochs and train
epochs = 20
for ep in range(epochs):
    for i in range(len(FP_data.file_name)):
        # Extract flight plan
        x_3D = FP_data[i]
        print(len(x_3D))
        ########################################
        # Extract flight tracks
        y_3D = FT_data[i]
        print(len(y_3D))
        ########################################
        # Weather data
        x_w = torch.randn(len(x_3D), 1, 20, 20) # random weather data

        train_dataset = [x_w, x_3D, y_3D]

        optimizer.zero_grad()
        # initiate LSTM hidden cell with 0
        model.hidden_cell = (torch.zeros(1, 1, model.lstm_hidden),
                             torch.zeros(1, 1, model.lstm_hidden))
        # input_seq = flight trajectory data + weather features
        y_pred = model(x_w, x_3D)
        # print(y_pred)
        single_loss = loss_function(y_pred, y_3D)
        single_loss.backward()
        optimizer.step()
    # pint each epoch loss
    print(f'epoch: {ep:1} loss: {single_loss.item():10.8f}')


