import numpy as np
import csv
import torch
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from custom_dataset import CustomDataset
from custom_dataset import ToTensor
import tqdm
import torch.nn as nn
from model import CONV_LSTM


root_dir = 'data/'
flight_data = CustomDataset(root_dir, ToTensor())

# train_model
model = CONV_LSTM() #customized Convolution and LSTM model
# set training epochs and train
epochs = 5

model.fit(flight_data, epochs, mode='Seq2Seq')



