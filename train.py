import numpy as np
import csv
import torch
from netCDF4 import Dataset
from datetime import datetime
from matplotlib import pyplot as plt
from custom_dataset import CustomDataset
from custom_dataset import ToTensor
import torch.nn as nn
from model import CONV_LSTM


# dev='cuda:0'
dev = 'cpu'
# root_dir = 'data/' TEST DATA
root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'
flight_data = CustomDataset(root_dir, ToTensor(), dev)
flight_data.validate_sets(underMin=100)

# train_model
paradigms = {0: 'Regression', 1: 'Seq2Seq'}
model = CONV_LSTM(paradigm=paradigms[1],device=dev) #customized Convolution and LSTM model
# set training epochs and train
epochs = 5
sttime = datetime.now()
print('START FIT: {}'.format(sttime))

model.fit(flight_data, epochs)

edtime = datetime.now()
print('END FIT: {}'.format(edtime))

model.save_model(epochs=epochs)


print('DONE: {}'.format(edtime-sttime))



