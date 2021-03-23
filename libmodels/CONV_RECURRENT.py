import torch
import tqdm
# from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.utils import Batch_norm_overtime as BatchNorm
from libmodels.Standalone_Self_Attention.attention import AttentionConv
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# customized Convolution and LSTM model
class CONV_RECURRENT(nn.Module):
    def __init__(self, paradigm='Seq2Seq', device='cpu', conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16,
                 dense_output=4, rnn= torch.nn.LSTM, rnn_layers=1, rnn_input=6, rnn_hidden=100, rnn_output=2,
                 attn='None',
                 optim:torch.optim=torch.optim.Adam, loss=torch.nn.MSELoss(), eptrained=0):
        super().__init__()
        # convolution layer for weather feature extraction prior to the RNN
        self.device = device
        self.paradigm = paradigm
        self.attntype = attn

        self.conv_input = conv_input
        self.conv_hidden = conv_hidden
        self.conv_output = conv_output
        self.dense_hidden = dense_hidden
        self.dense_output = dense_output

        self.rnn_type = rnn
        self.rnn_layers = rnn_layers
        self.rnn_input = rnn_input
        self.rnn_hidden = rnn_hidden
        self.rnn_output = rnn_output
        self.hidden_cell = None

        self.convs = nn.ModuleList()
        if self.attntype == 'replace':
            #TODO VALID IMPL
            self.convs.append(AttentionConv(in_channels=self.conv_input, out_channels=self.conv_hidden, kernel_size=4,
                                            stride=1, padding=0, groups=1, bias=False))
            self.convs.append(AttentionConv(in_channels=self.conv_hidden, out_channels=self.conv_output, kernel_size=6,
                                            stride=2, padding=0, groups=1, bias=False))
        else:
            self.convs.append(torch.nn.Conv2d(self.conv_input, self.conv_hidden, kernel_size=6, stride=2))
            self.convs.append(torch.nn.Conv2d(self.conv_hidden, self.conv_output, kernel_size=3, stride=2))
        if self.attntype == 'after':
            self.convs.append(AttentionConv(in_channels=4,out_channels=4,kernel_size=3,stride=1,padding=0,groups=1,bias=False))


        if self.rnn_type == indrnn:
            self.fc1 = torch.nn.Linear(self.conv_output*9, self.rnn_hidden)
            self.fc2 = torch.nn.Linear(self.rnn_hidden, self.rnn_hidden-2)
        elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
            # 64 input features, 16 output features (see sizing flow below)
            self.fc1 = torch.nn.Linear(self.conv_output * 9, self.dense_hidden)
            # 16 input features, 4 output features (see sizing flow below)
            self.fc2 = torch.nn.Linear(self.dense_hidden, self.rnn_input-2)

        ################################################################
        # IndRNN model
        self.rnns = nn.ModuleList()

        for i in range(self.rnn_layers):
            if self.rnn_type == indrnn:
                self.rnns.append(self.rnn_type(hidden_size=self.rnn_hidden))
                self.rnns.append(BatchNorm(hidden_size=self.rnn_hidden, seq_len=-1))
            elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
                self.rnns.append(self.rnn_type(input_size=self.rnn_input, hidden_size=self.rnn_hidden))
            # gru/lstm(input_size, hidden_size, num_layers)
            # indrnn(hidden_size)
            # indrnns[i].indrnn_cell initialized on uniform distr. Set to coordinates in training

        if self.rnn_type == torch.nn.LSTM:
            # h_,c_ each of size (num_layers*num_directions, batch_size, hidden_size)
            self.hidden_cell = (torch.zeros((self.rnn_layers * 1, 1, self.rnn_hidden)),
                                torch.zeros((self.rnn_layers * 1, 1, self.rnn_hidden)))
        elif self.rnn_type == torch.nn.GRU:
            # h_ of size (num_layers*num_directions, batch_size, hidden_size)
            self.hidden_cell = torch.zeros((self.rnn_layers * 1, 1, self.rnn_hidden))
        elif self.rnn_type == indrnn:
            # indrnn does not use an external cell state
            pass

        self.linear = nn.Linear(self.rnn_hidden, self.rnn_output)

        if self.device.__contains__('cuda'):
            self.cuda(self.device)

        if not issubclass(optim, torch.optim.Optimizer):
            optim = type(optim)

        self.optimizer = optim(self.parameters())
        self.loss_function = loss
        self.epochs_trained = eptrained

        self.struct_dict = {'class': str(type(self)).split('\'')[1],
                            'device': self.device, 'paradigm': self.paradigm,
                            'conv_input': self.conv_input, 'conv_hidden': self.conv_hidden,
                            'conv_output': self.conv_output, 'attntype': self.attntype,
                            'dense_hidden': self.dense_hidden, 'dense_output': self.dense_output,
                            'rnn_type': self.rnn_type, 'rnn_layers': self.rnn_layers,
                            'rnn_input': self.rnn_input, 'rnn_hidden': self.rnn_hidden,
                            'rnn_output': self.rnn_output, 'hidden_cell': self.hidden_cell,
                            'loss_fn': self.loss_function, 'optim': type(self.optimizer)}

    def forward(self, x_w, x_t):
        # convolution apply first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)
        tmp = x_w.view(-1,1,20,20)
        for i in range(len(self.convs)):
            tmp = F.relu(self.convs[i](tmp))
        '''
        x_conv_1 = F.relu(self.conv_1(x_w.view(-1,1,20,20)))
        x_conv_2 = F.relu(self.conv_2(x_conv_1))
        '''

        # Flatten the convolution layer output
        x_conv_output = tmp.view(-1, self.conv_output * 9)
        x_fc_1 = F.relu(self.fc1(x_conv_output))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 256) to (1, 64)
        x_fc_2 = F.relu(self.fc2(x_fc_1))

        #############################################################
        # input_seq = flight trajectory data + weather features
        # shape: [seq_len, batch_size, input_features]
        rnn_input_seq = torch.cat((x_fc_2.view(x_t.size(0),x_t.size(1),-1), x_t,), -1)
        rnnout = rnn_input_seq
        for i in range(len(self.rnns)):
            if self.rnn_type == indrnn:
                rnnout = self.rnns[i](rnnout)
            elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
                rnnout, self.hidden_cell = self.rnns[i](rnnout)

        # feed input_seq into LSTM model
        #lstm_out, self.hidden_cell = self.lstm(lstm_input_seq, self.hidden_cell)

        # TODO: Dimension expansion (eq. 2l in Pang, Xu, Liu)
        predictions = self.linear(rnnout)
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
        recurrence = str(type(self.rnns[0])).split('\'')[1].split('.')[-1]
        opt = str(self.optimizer.__class__).split('\'')[1].split('.')[-1]
        convs = 'CONV-'
        if self.attntype == 'after':
            convs = convs + 'SA-'
        model_name = '{}{}-OPT{}-LOSS{}-EPOCHS{}-BATCH{}-RNN{}_{}_{}'.format(convs, recurrence, opt, self.loss_function,
                                                                                self.epochs_trained,
                                                                                batch_size, self.rnn_input,
                                                                                self.rnn_hidden, self.rnn_output)
        return model_name