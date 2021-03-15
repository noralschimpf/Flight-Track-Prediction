import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class CONV_GRU(nn.Module):
    def __init__(self, paradigm='Seq2Seq', device='cpu', conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16,
                 dense_output=4, gru_input=6, gru_hidden=100, gru_output=2):
        # conv and gru input and output parameters can be customized
        super().__init__()
        # convolution layer for weather feature extraction prior to the GRU
        self.device = device
        self.paradigm = paradigm

        self.conv_input = conv_input
        self.conv_hidden = conv_hidden
        self.conv_output = conv_output
        self.dense_hidden = dense_hidden
        self.dense_output = dense_output

        self.gru_input = gru_input
        self.gru_hidden = gru_hidden
        self.gru_output = gru_output

        self.conv_1 = torch.nn.Conv2d(self.conv_input, self.conv_hidden, kernel_size=6, stride=2)
        self.conv_2 = torch.nn.Conv2d(self.conv_hidden, self.conv_output, kernel_size=3, stride=2)
        # 64 input features, 16 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.conv_output * 9, self.dense_hidden)
        # 16 input features, 4 output features (see sizing flow below)
        self.fc2 = torch.nn.Linear(self.dense_hidden, self.dense_output)

        ################################################################
        # GRU model
        # concatenate flight trajectory input with weather features
        self.gru = nn.GRU(self.gru_input, self.gru_hidden, batch_first=True)
        self.linear = nn.Linear(self.gru_hidden, self.gru_output)

        # hidden cell dims: [batch_size][num_layers*num_directions][hidden_size]
        self.hidden_cell = torch.zeros(1, 1, self.gru_hidden)

        if self.device.__contains__('cuda'):
            self.cuda(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

        self.struct_dict = {'class': str(self.__class__).split('\'')[1], 'device': self.device, 'paradigm': self.paradigm,
                            'conv_input': self.conv_input, 'conv_hidden': self.conv_hidden,
                            'conv_output': self.conv_output,
                            'dense_hidden': self.dense_hidden, 'dense_output': self.dense_output,
                            'gru_input': self.gru_input, 'gru_hidden': self.gru_hidden,
                            'gru_output': self.gru_output,
                            'loss_fn': self.loss_function, 'optim': self.optimizer}

    def forward(self, x_w, x_t):
        # convolution apply first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)

        x_conv_1 = F.relu(self.conv_1(x_w.view(-1, 1, 20, 20)))
        x_conv_2 = F.relu(self.conv_2(x_conv_1))

        # Flatten the convolution layer output
        x_conv_output = x_conv_2.view(-1, self.conv_output * 9)
        x_fc_1 = F.relu(self.fc1(x_conv_output))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 256) to (1, 64)
        x_fc_2 = F.relu(self.fc2(x_fc_1))

        #############################################################
        # input_seq = flight trajectory data + weather features
        # shape: [batch_size, seq_len, features] - note: batch_first=True
        gru_input_seq = torch.cat((x_fc_2.view(x_t.size(0), -1, 4), x_t), -1)

        # feed input_seq into LSTM model
        gru_out, self.hidden_cell = self.gru(gru_input_seq, self.hidden_cell)

        # TODO: Dimension expansion (eq. 2l in Pang, Xu, Liu)
        predictions = self.linear(gru_out)
        return predictions

    def save_model(self, epochs, batch_size: str, model_name: str = None):
        if model_name == None:
            model_name = self.model_name(epochs=epochs, batch_size=batch_size)
        model_path = 'Models/{}/{}'.format(model_name, model_name)
        while os.path.isfile(model_path):
            choice = input("Model Exists:\n1: Replace\n2: New Model\n")
            if choice == '1':
                break
            elif choice == '2':
                name = input("Enter model name\n")
                model_path = 'Models/' + name
        if not os.path.isdir('Models/{}'.format(model_name)):
            os.mkdir('Models/{}'.format(model_name))
        torch.save({'struct_dict': self.struct_dict, 'state_dict': self.state_dict()}, model_path)

    def model_name(self, epochs: int = 500, batch_size: int = 1):
        opt = str(self.optimizer.__class__).split('\'')[1].split('.')[-1]
        model_name = 'CONV-GRU-OPT{}-LOSS{}-EPOCHS{}-BATCH{}-GRU{}_{}_{}'.format(opt, self.loss_function, epochs,
                                                                                  batch_size, self.gru_input,
                                                                                  self.gru_hidden, self.gru_output)
        return model_name