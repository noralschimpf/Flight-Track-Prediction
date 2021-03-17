import torch
import tqdm
# from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# customized Convolution and LSTM model
#TODO: Replace LSTM with IndRNN
class CONV_INDRNN(nn.Module):
    def __init__(self, paradigm='Seq2Seq', device='cpu', conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16,
                 dense_output=4, rnn_input=6, rnn_hidden=100, rnn_output=2,
                 optim:torch.optim=torch.optim.Adam, loss=torch.nn.MSELoss(), eptrained=0):
        super().__init__()
        # convolution layer for weather feature extraction prior to the RNN
        self.device = device
        self.paradigm = paradigm

        self.conv_input = conv_input
        self.conv_hidden = conv_hidden
        self.conv_output = conv_output
        self.dense_hidden = dense_hidden
        self.dense_output = dense_output

        self.rnn_input = rnn_input
        self.rnn_hidden = rnn_hidden
        self.rnn_output = rnn_output

        self.conv_1 = torch.nn.Conv2d(self.conv_input, self.conv_hidden, kernel_size=6, stride=2)
        self.conv_2 = torch.nn.Conv2d(self.conv_hidden, self.conv_output, kernel_size=3, stride=2)
        # 64 input features, 16 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.conv_output * 9, self.dense_hidden)
        # 16 input features, 4 output features (see sizing flow below)
        self.fc2 = torch.nn.Linear(self.dense_hidden, self.dense_output)

        ################################################################
        # IndRNN model #TODO: Match layering technique to other models

        if self.device.__contains__('cuda'):
            indrnn_base = cuda_indrnn
        else:
            indrnn_base = indrnn
        self.rnns = nn.ModuleList()

        for i in range(1+1):
            self.rnns.append(indrnn_base(hidden_size=self.rnn_input))
            # indrnns[i].indrnn_cell initialized on uniform distr. Set to coordinates in training


        self.linear = nn.Linear(self.rnn_input, self.rnn_output)
        # IndRNN Cell State auto-initializes

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
                            'rnn_input': self.rnn_input, 'rnn_hidden': self.rnn_hidden,
                            'rnn_output': self.rnn_output,
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
        # shape: [seq_len, batch_size, input_features]
        rnn_input_seq = torch.cat((x_fc_2.view(x_t.size(0), x_t.size(1), self.rnn_input - 2), x_t), -1)
        rnnouts = {}
        rnnouts['rnn-out-1'] = rnn_input_seq
        for i in range(len(self.rnns)):
            rnnouts['rnn-out{}'.format(i)] = self.rnns[i](rnnouts['rnn-out{}'.format(i - 1)])

        # feed input_seq into LSTM model
        #lstm_out, self.hidden_cell = self.lstm(lstm_input_seq, self.hidden_cell)

        # TODO: Dimension expansion (eq. 2l in Pang, Xu, Liu)
        predictions = self.linear(rnnouts['rnn-out{}'.format(len(self.rnns) - 1)])
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

    #TODO: Generalize for all models
    #TODO: epochs from self.epochs_trained
    def model_name(self, batch_size: int = 1):
        recurrence = str(type(self.rnns[0])).split('\'')[1].split('.')[-1]
        opt = str(self.optimizer.__class__).split('\'')[1].split('.')[-1]
        model_name = 'CONV-{}-OPT{}-LOSS{}-EPOCHS{}-BATCH{}-LSTM{}_{}_{}'.format(recurrence, opt, self.loss_function,
                                                                                self.epochs_trained,
                                                                                batch_size, self.rnn_input,
                                                                                self.rnn_hidden, self.rnn_output)
        return model_name