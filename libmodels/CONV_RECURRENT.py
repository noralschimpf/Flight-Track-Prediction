import torch
import tqdm
#from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.utils import Batch_norm_overtime as BatchNorm
from libmodels.multiheaded_attention import MultiHeadedAttention as MHA
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# customized Convolution and LSTM model
class CONV_RECURRENT(nn.Module):
    def __init__(self, paradigm='Seq2Seq', device='cpu', cube_height=1, conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16,
                 dense_output=4, rnn= torch.nn.LSTM, rnn_layers=1, rnn_input=6, rnn_hidden=100, rnn_output=3,
                 attn='None', batch_size=1, droprate = .2, num_features: int = 1,
                 optim:torch.optim=torch.optim.Adam, loss=torch.nn.MSELoss(), eptrained=0):
        super().__init__()
        # convolution layer for weather feature extraction prior to the RNN
        self.device = device
        self.paradigm = paradigm
        self.attntype = attn
        self.batch_size = batch_size
        self.droprate = droprate
        self.num_features = num_features
        self.cube_height = cube_height

        self.conv_input = conv_input
        self.conv_hidden = conv_hidden
        self.conv_output = conv_output
        self.dense_hidden = dense_hidden

        self.rnn_type = rnn
        self.rnn_layers = rnn_layers
        self.rnn_input = rnn_input
        self.rnn_hidden = rnn_hidden
        self.rnn_output = rnn_output
        self.hidden_cell = None


        self.convs = nn.ModuleList()
        for i in range(self.num_features):
            extractor = nn.ModuleList()
            if self.attntype == 'replace':
                extractor.append(MHA(d_model=128, num_heads=2, p=0, d_input=self.conv_input * self.cube_height * 400))
                extractor.append(MHA(d_model=36, num_heads=4, p=0, d_input=128))
                extractor.append(torch.nn.Dropout(self.droprate))
                extractor.append(MHA(d_model=self.conv_output*9,num_heads=3,p=0,d_input=36))

            elif self.attntype == 'after':
                extractor.append(torch.nn.Conv3d(self.conv_input, self.conv_hidden, kernel_size=(self.cube_height, 6, 6), stride=2))
                extractor.append(torch.nn.Conv3d(self.conv_hidden, self.conv_output, kernel_size=(1,3,3), stride=2))
                extractor.append(torch.nn.Flatten(1,-1))
                extractor.append(torch.nn.Dropout(self.droprate))
                extractor.append(MHA(d_model=self.conv_output*9, num_heads=3, p=0, d_input=36))

            else:
                extractor.append(torch.nn.Conv3d(self.conv_input, self.conv_hidden, kernel_size=(self.cube_height, 6, 6), stride=2))
                extractor.append(torch.nn.Conv3d(self.conv_hidden, self.conv_output, kernel_size=(1,3,3), stride=2))
                extractor.append(torch.nn.Dropout(self.droprate))
                extractor.append(torch.nn.Conv3d(self.conv_output, self.conv_output, kernel_size=(1,1,1), stride=1))
            self.convs.append(extractor)

        if self.rnn_type == indrnn or self.rnn_type == cuda_indrnn:
            self.fc1 = torch.nn.Linear(self.num_features * self.conv_output*9, self.dense_hidden)
            self.fc2 = torch.nn.Linear(self.dense_hidden, self.rnn_hidden-3)
        elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
            # 64 input features, 16 output features (see sizing flow below)
            self.fc1 = torch.nn.Linear(self.num_features * self.conv_output * 9, self.dense_hidden)
            # 16 input features, 4 output features (see sizing flow below)
            self.fc2 = torch.nn.Linear(self.dense_hidden, self.rnn_input-3)

        ################################################################
        # IndRNN model
        self.rnns = nn.ModuleList()

        for i in range(self.rnn_layers):
            if self.rnn_type == indrnn or self.rnn_type == cuda_indrnn:
                if i == 0:
                    self.rnns.append(torch.nn.Dropout(self.droprate))
                self.rnns.append(self.rnn_type(hidden_size=self.rnn_hidden))
                self.rnns.append(BatchNorm(hidden_size=self.rnn_hidden, seq_len=-1))
                if not i == self.rnn_layers:
                    self.rnns.append(torch.nn.Dropout(self.droprate))
            elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
                if i == 0:
                    self.rnns.append(torch.nn.Dropout(self.droprate))
                    self.rnns.append(self.rnn_type(input_size=self.rnn_input, hidden_size=self.rnn_hidden,
                                                   num_layers=self.rnn_layers, dropout=droprate))
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
        elif self.rnn_type == indrnn or self.rnn_type == cuda_indrnn:
            # h_ of size (num_lay, batch, hidden_size)
            self.hidden_cell = torch.zeros((self.rnn_layers*1, 1, self.rnn_hidden))


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
                            'dense_hidden': self.dense_hidden, 'num_features': self.num_features,
                            'rnn_type': self.rnn_type, 'rnn_layers': self.rnn_layers,
                            'rnn_input': self.rnn_input, 'rnn_hidden': self.rnn_hidden,
                            'rnn_output': self.rnn_output, 'hidden_cell': self.hidden_cell, 'droprate': self.droprate,
                            'loss_fn': self.loss_function, 'optim': type(self.optimizer)}

    def forward(self, x_w, x_t):
        # convolution apply first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)
        #TODO: VALIDATE FORWARD PASS FOR ALL MODEL TYPES
        if isinstance(self.convs[0][0], MHA):
            conv_outs = {'cnv-1':x_w.view(x_w.shape[0], self.num_features, -1, self.cube_height*400)}
        else:
            tmp = x_w.view(-1, self.num_features, self.conv_input, x_w.shape[-3], x_w.shape[-2], x_w.shape[-1])
            conv_outs = {'cnv-1': tmp}
        for i in range(len(self.convs[0])):
            pastkey = 'cnv{}'.format(i - 1)
            curkey = 'cnv{}'.format(i)
            if isinstance(self.convs[0][i], torch.nn.Conv2d) or isinstance(self.convs[0][i],torch.nn.Conv3d):
                seqlen = conv_outs[pastkey].shape[0]
                channels = self.convs[0][i].out_channels
                new_depth = int((conv_outs[pastkey].shape[-3] - self.convs[0][i].kernel_size[0])/self.convs[0][i].stride[0])+1
                new_cube = int((conv_outs[pastkey].shape[-1] - self.convs[0][i].kernel_size[-1]) / self.convs[0][i].stride[0])+1
                conv_outs[curkey] = torch.zeros((seqlen, self.num_features, channels, new_depth, new_cube, new_cube), device=self.device)
                for f in range(self.num_features):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])
                conv_outs[curkey] = torch.relu(conv_outs[curkey])
            elif isinstance(self.convs[0][i],torch.nn.Dropout):
                conv_outs[curkey] = torch.zeros_like(conv_outs[pastkey], device=self.device)
                for f in range(self.num_features):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])
            elif isinstance(self.convs[0][i], MHA):
                seqlen = conv_outs[pastkey].shape[0]
                batchsize = conv_outs[pastkey].shape[2]
                features_out = self.convs[0][i].d_model
                conv_outs[curkey] = torch.zeros((seqlen, self.num_features, batchsize, features_out), device=self.device)
                for f in range(self.num_features):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])


        # Flatten the convolution layer output
        outkey = 'cnv{}'.format(len(conv_outs)-2)
        x_conv_output = conv_outs[outkey].view(-1, self.fc1.in_features)
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
            if isinstance(self.rnns[i], indrnn) or isinstance(self.rnns[i], cuda_indrnn):
                rnnlay = 0
                for j in range(len(self.rnns[:i])):
                    if isinstance(self.rnns[j], indrnn) or isinstance(self.rnns[j], cuda_indrnn): rnnlay += 1
                rnnout = self.rnns[i](rnnout, self.hidden_cell[rnnlay])
            elif isinstance(self.rnns[i], BatchNorm) or isinstance(self.rnns[i], torch.nn.Dropout):
                rnnout = self.rnns[i](rnnout)
            elif isinstance(self.rnns[i], torch.nn.LSTM) or isinstance(self.rnns[i], torch.nn.GRU):
                rnnout, self.hidden_cell = self.rnns[i](rnnout)

        # feed input_seq into LSTM model
        #lstm_out, self.hidden_cell = self.lstm(lstm_input_seq, self.hidden_cell)

        # TODO: Dimension expansion (eq. 2l in Pang, Xu, Liu)
        predictions = self.linear(rnnout)
        return predictions

    def init_hidden_cell(self, tns_coords: torch.Tensor):
        # hidden cell (h_, c_) sizes: [num_layers*num_directions, batch_size, hidden_size]
        if self.rnn_type == torch.nn.LSTM:
            self.hidden_cell = (
                tns_coords.repeat(self.rnn_layers, 1, 1),
                tns_coords.repeat(self.rnn_layers, 1, 1))
        elif self.rnn_type == torch.nn.GRU:
            self.hidden_cell = torch.cat((
                tns_coords.repeat(self.rnn_layers, 1, 1),
            ))
        elif self.rnn_type == indrnn or self.rnn_type == cuda_indrnn:
            #h_ = [layers, batch, hidden_size]
            self.hidden_cell = torch.cat((tns_coords.repeat(self.rnn_layers, 1, 1),))


    def save_model(self, model_name: str = None, override: bool = False):
        if model_name == None:
            model_name = self.model_name()
        model_path = 'Models/{}/{}'.format(model_name, model_name)
        while os.path.isfile(model_path) and not override:
            choice = input("Model Exists:\n1: Replace\n2: New Model\n")
            if choice == '1':
                break
            elif choice == '2':
                name = input("Enter model name\n")
                model_path = 'Models/' + name
        if not os.path.isdir('Models/{}'.format(model_name)):
            os.mkdir('Models/{}'.format(model_name))
        torch.save({'struct_dict': self.struct_dict, 'state_dict': self.state_dict(),
                    'opt_dict': self.optimizer.state_dict(), 'epochs_trained': self.epochs_trained,
                    'batch_size': self.batch_size}, model_path)

    def model_name(self):
        recurrence = str(self.rnn_type).split('\'')[1].split('.')[-1]
        recurrence = recurrence + '{}lay'.format(self.rnn_layers)
        opt = str(self.optimizer.__class__).split('\'')[1].split('.')[-1]
        if self.attntype == 'None':
            convs = 'CONV{}-'.format(self.num_features)
        if self.attntype == 'after':
            convs = 'SAA{}-'.format(self.num_features)
        elif self.attntype == 'replace':
            convs = 'SAR{}-'.format(self.num_features)
        model_name = '{}{}-OPT{}-LOSS{}-EPOCHS{}-BATCH{}-RNN{}_{}_{}'.format(convs, recurrence, opt, self.loss_function,
                                                                                self.epochs_trained,
                                                                                self.batch_size, self.rnn_input,
                                                                                self.rnn_hidden, self.rnn_output)
        return model_name