import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# customized Convolution and LSTM model
class CONV_LSTM(nn.Module):
    def __init__(self, paradigm='Seq2Seq', device='cpu', conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16,
                 dense_output=4, lstm_input=6, lstm_hidden=100, lstm_output=2,
                 optim: torch.optim = torch.optim.Adam, loss=torch.nn.MSELoss(), eptrained = 0):
        # conv and lstm input and output parameters can be customized
        super().__init__()
        # convolution layer for weather feature extraction prior to the LSTM
        self.device = device
        self.paradigm = paradigm

        self.conv_input = conv_input
        self.conv_hidden = conv_hidden
        self.conv_output = conv_output
        self.dense_hidden = dense_hidden
        self.dense_output = dense_output

        self.lstm_input = lstm_input
        self.lstm_hidden = lstm_hidden
        self.lstm_output = lstm_output

        self.conv_1 = torch.nn.Conv2d(self.conv_input, self.conv_hidden, kernel_size=6, stride=2)
        self.conv_2 = torch.nn.Conv2d(self.conv_hidden, self.conv_output, kernel_size=3, stride=2)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 64 input features, 16 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.conv_output * 9, self.dense_hidden)
        # 16 input features, 4 output features (see sizing flow below)
        self.fc2 = torch.nn.Linear(self.dense_hidden, self.dense_output)

        ################################################################
        # LSTM model
        # concatenate flight trajectory input with weather features
        self.lstm = nn.LSTM(self.lstm_input, self.lstm_hidden)
        self.linear = nn.Linear(self.lstm_hidden, self.lstm_output)
        # h_0, c_0 sizes (num_layers*num_directions, batch, hidden_size)
        self.hidden_cell = (torch.zeros(1, 1, self.lstm_hidden), torch.zeros(1, 1, self.lstm_hidden))

        if self.device.__contains__('cuda'):
            self.cuda(self.device)

        if not issubclass(optim, torch.optim.Optimizer):
            optim = type(optim)
        self.optimizer = optim(self.parameters())
        self.loss_function = loss
        self.epochs_trained = eptrained
        self.struct_dict = {'class': str(self.__class__).split('\'')[1],
                            'device': self.device, 'paradigm': self.paradigm,
                            'conv_input': self.conv_input, 'conv_hidden': self.conv_hidden,
                            'conv_output': self.conv_output,
                            'dense_hidden': self.dense_hidden, 'dense_output': self.dense_output,
                            'lstm_input': self.lstm_input, 'lstm_hidden': self.lstm_hidden,
                            'lstm_output': self.lstm_output,
                            'loss_fn': self.loss_function, 'optim': type(self.optimizer)}

    def forward(self, x_w, x_t):
        # convolution apply first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)
        x_conv_1 = F.relu(self.conv_1(x_w.view(-1,1,20,20)))
        x_conv_2 = F.relu(self.conv_2(x_conv_1))

        # Flatten the convolution layer output
        x_conv_output = x_conv_2.view(-1, self.conv_output * 9)
        x_fc_1 = F.relu(self.fc1(x_conv_output))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 256) to (1, 64)
        x_fc_2 = F.relu(self.fc2(x_fc_1))

        #############################################################
        # input_seq = flight trajectory data + weather features
        # shape: [seq_len, batch_size, features] - note: batch_first=False
        lstm_input_seq = torch.cat((x_fc_2.view(x_t.size(0), x_t.size(1), self.lstm_input - 2), x_t), -1)

        # feed input_seq into LSTM model
        lstm_out, self.hidden_cell = self.lstm(lstm_input_seq, self.hidden_cell)

        # TODO: Dimension expansion (eq. 2l in Pang, Xu, Liu)
        predictions = self.linear(lstm_out)
        return predictions

    def save_model(self, batch_size: str, model_name: str = None):
        if model_name == None:
            model_name = self.model_name(batch_size=batch_size)
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
        torch.save({'struct_dict': self.struct_dict, 'state_dict': self.state_dict(),
                    'opt_dict': self.optimizer.state_dict(), 'epochs_trained': self.epochs_trained}, model_path)

    def model_name(self, batch_size: int = 1):
        opt = str(self.optimizer.__class__).split('\'')[1].split('.')[-1]
        model_name = 'CONV-LSTM-OPT{}-LOSS{}-EPOCHS{}-BATCH{}-LSTM{}_{}_{}'.format(opt, self.loss_function,
                                                                                self.epochs_trained,
                                                                                batch_size, self.lstm_input,
                                                                                self.lstm_hidden, self.lstm_output)
        return model_name