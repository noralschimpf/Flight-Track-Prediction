import torch, torch.nn as nn, torch.nn.functional as F
import os, inspect

from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.utils import Batch_norm_overtime as BatchNorm
from libmodels.multiheaded_attention import MultiHeadedAttention as MHA

# customized Convolution and LSTM model
class CONV_RECURRENT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        initdict = self.__dict__.copy()

        ################################################################
        # Store Model Configuration as attributes
        for key in config:
            if key == 'ConvCh':
                self.conv_input = config[key][0]
                self.conv_hidden = config[key][1]
                self.conv_output = config[key][2]
            else: self.__setattr__(key,config[key])
        if config['batchnorm'] == 'None':
            self.bn = False
            self.bn_af = False
        elif config['batchnorm'] == 'simple':
            self.bn = True
            self.bn_af = False
        elif config['batchnorm'] == 'learn':
            self.bn = True
            self.bn_af = True
        self.classname = str(type(self)).split('\'')[1]
        structdict = self.__dict__.copy()


        ################################################################
        # Initialize Model
        self.hidden_cell = None

        self.convs = nn.ModuleList()
        for i in range(len(self.features)):
            extractor = nn.ModuleList()
            if self.attntype == 'replace':
                extractor.append(MHA(d_model=64*self.conv_hidden, num_heads=self.conv_hidden, p=0, d_input=self.conv_input * self.cube_height * 400))
                if self.bn: extractor.append(torch.nn.BatchNorm1d(64*self.conv_hidden, affine=self.bn_af))
                extractor.append(MHA(d_model=9*self.conv_output, num_heads=self.conv_output, p=0, d_input=64*self.conv_hidden))
                if self.bn: extractor.append(torch.nn.BatchNorm1d(9*self.conv_output, affine=self.bn_af))
                extractor.append(torch.nn.Dropout(self.droprate))
                extractor.append(MHA(d_model=self.conv_output*9,num_heads=self.conv_output,p=0,d_input=9*self.conv_output))
                if self.bn: extractor.append(torch.nn.BatchNorm1d(self.conv_output*9, affine=self.bn_af))

            elif self.attntype == 'after':
                extractor.append(torch.nn.Conv3d(self.conv_input, self.conv_hidden, kernel_size=(self.cube_height, 6, 6), stride=2))
                if self.bn: extractor.append(torch.nn.BatchNorm3d(self.conv_hidden, affine=self.bn_af))
                extractor.append(torch.nn.Conv3d(self.conv_hidden, self.conv_output, kernel_size=(1,3,3), stride=2))
                if self.bn: extractor.append(torch.nn.BatchNorm3d(self.conv_output, affine=self.bn_af))
                extractor.append(torch.nn.Flatten(1))
                extractor.append(torch.nn.Dropout(self.droprate))
                extractor.append(MHA(d_input=self.conv_output*9, num_heads=3, p=0, d_model=self.conv_output*9))
                if self.bn: extractor.append(torch.nn.BatchNorm1d(self.conv_output*9, affine=self.bn_af))

            else:
                extractor.append(torch.nn.Conv3d(self.conv_input, self.conv_hidden, kernel_size=(self.cube_height, 6, 6), stride=2))
                if self.bn: extractor.append(torch.nn.BatchNorm3d(self.conv_hidden, affine=self.bn_af))
                extractor.append(torch.nn.Conv3d(self.conv_hidden, self.conv_output, kernel_size=(1,3,3), stride=2))
                if self.bn: extractor.append(torch.nn.BatchNorm3d(self.conv_output, affine=self.bn_af))
                extractor.append(torch.nn.Dropout(self.droprate))
                extractor.append(torch.nn.Conv3d(self.conv_output, self.conv_output, kernel_size=(1,1,1), stride=1))
                if self.bn: extractor.append(torch.nn.BatchNorm3d(self.conv_output, affine=self.bn_af))
            self.convs.append(extractor)

        self.fc = torch.nn.ModuleList()
        self.fc.append(torch.nn.Linear(len(self.features)*self.conv_output*9, self.dense_hidden[0]))
        for i in range(len(self.dense_hidden)-1):
            self.fc.append(torch.nn.Linear(self.dense_hidden[i],self.dense_hidden[i+1]))
            if self.bn: self.fc.append(torch.nn.BatchNorm1d(self.dense_hidden[i+1], affine=self.bn_af))
        if self.rnn_type == indrnn or self.rnn_type == cuda_indrnn:
            self.fc.append(torch.nn.Linear(self.dense_hidden[-1], self.rnn_hidden-3))
            if self.bn: self.fc.append(torch.nn.BatchNorm1d(self.rnn_hidden-3, affine=self.bn_af))
        elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
            self.fc.append(torch.nn.Linear(self.dense_hidden[-1], self.rnn_input-3))
            if self.bn: self.fc.append(torch.nn.BatchNorm1d(self.rnn_input - 3, affine=self.bn_af))

        ################################################################
        # IndRNN model
        self.rnns = nn.ModuleList()

        for i in range(self.rnn_layers):
            if i == 0:
                lstm_insize = self.rnn_input
                self.rnns.append(torch.nn.Dropout(self.droprate))
            else: lstm_insize = self.rnn_hidden

            if self.rnn_type == indrnn or self.rnn_type == cuda_indrnn: self.rnns.append(self.rnn_type(hidden_size=self.rnn_hidden))
            elif self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU:
                self.rnns.append(self.rnn_type(input_size=lstm_insize, hidden_size=self.rnn_hidden))
            #if self.bn: self.rnns.append(BatchNorm(hidden_size=self.rnn_hidden, seq_len=-1))
            if self.bn: self.rnns.append(torch.nn.BatchNorm1d(self.rnn_hidden, affine=self.bn_af))
            if not i == self.rnn_layers:
                self.rnns.append(torch.nn.Dropout(self.droprate))

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

        if len(self.rnns) == 0 and (self.rnn_type == indrnn or self.rnn_type == cuda_indrnn):
            self.linear = nn.Linear(self.rnn_hidden, self.rnn_output)
        elif len(self.rnns) == 0 and (self.rnn_type == torch.nn.LSTM or self.rnn_type == torch.nn.GRU):
            self.linear = nn.Linear(self.rnn_input, self.rnn_output)
        else:
            self.linear = nn.Linear(self.rnn_hidden, self.rnn_output)

        if self.device.__contains__('cuda'):
            self.cuda(int(self.device.split(':')[1]))
            self.device = '{}:{}'.format(self.fc[0].bias.device.type, self.fc[0].bias.device.index)
            structdict['device'] = self.device

        ################################################################
        # Initialize Loss Function
        self.loss_function = self.loss_function()

        ################################################################
        # Configure optimizer and optimizer scheduling
        args, varargs, keywords, defaults = inspect.getargspec(config['optimizer'].__init__)
        if 'lr' in args and 'weight_decay' in args:
            if 'momentum' in args and 'nesterov' in args:
                self.optimizer = config['optimizer'](self.parameters(), lr=config['lr'], weight_decay=config['weight_reg'],
                                                momentum=config['momentum'],
                                                nesterov=config['nesterov'])
            else:
                self.optimizer = config['optimizer'](self.parameters(), lr=config['lr'], weight_decay=config['weight_reg'])
        self.optim = type(self.optimizer); structdict['optimizer'] = self.optim

        if 'decay_step' in config and 'decay_gamma' in config:
            self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=config['decay_step'], gamma=config['decay_gamma'])

        self.epochs_trained = 0

        self.struct_dict = {x: structdict[x] for x in (set(structdict) - set(initdict))}

        print(self)
        print(self.optimizer)



    def forward(self, x_w, x_t):
        # apply convolution first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)
        #TODO: VALIDATE FORWARD PASS FOR ALL MODEL TYPES
        if isinstance(self.convs[0][0], MHA):
            conv_outs = {'cnv-1':x_w.view(x_w.shape[0], len(self.features), -1, self.cube_height*400)}
        else:
            tmp = x_w.view(-1, len(self.features), self.conv_input, x_w.shape[-3], x_w.shape[-2], x_w.shape[-1])
            conv_outs = {'cnv-1': tmp}
        for i in range(len(self.convs[0])):
            pastkey = 'cnv{}'.format(i - 1)
            curkey = 'cnv{}'.format(i)
            if isinstance(self.convs[0][i], torch.nn.Conv2d) or isinstance(self.convs[0][i],torch.nn.Conv3d):
                seqlen = conv_outs[pastkey].shape[0]
                channels = self.convs[0][i].out_channels
                new_depth = int((conv_outs[pastkey].shape[-3] - self.convs[0][i].kernel_size[0])/self.convs[0][i].stride[0])+1
                new_cube = int((conv_outs[pastkey].shape[-1] - self.convs[0][i].kernel_size[-1]) / self.convs[0][i].stride[0])+1
                conv_outs[curkey] = torch.zeros((seqlen, len(self.features), channels, new_depth, new_cube, new_cube), device=self.device)
                for f in range(len(self.features)):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])
                conv_outs[curkey] = torch.relu(conv_outs[curkey])
            elif isinstance(self.convs[0][i],torch.nn.Dropout) or isinstance(self.convs[0][i], torch.nn.BatchNorm3d):
                conv_outs[curkey] = torch.zeros_like(conv_outs[pastkey], device=self.device)
                for f in range(len(self.features)):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])
            if isinstance(self.convs[0][i], torch.nn.BatchNorm1d):
                conv_outs[curkey] = torch.zeros_like(conv_outs[pastkey], device=self.device)
                if self.attntype == 'replace':
                    for f in range(len(self.features)):
                        conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:, f].permute(1,2,0)).permute(2,0,1)
                else:
                    for f in range(len(self.features)):
                        conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f].permute(1,2,0)).permute(2,0,1)
            elif isinstance(self.convs[0][i], torch.nn.Flatten):
                conv_outs[curkey] = torch.zeros_like(conv_outs[pastkey])
                conv_outs[curkey] = conv_outs[curkey].reshape((conv_outs[curkey].shape[0], conv_outs[curkey].shape[1],
                                                               -1))
                for f in range(len(self.features)):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])
            elif isinstance(self.convs[0][i], MHA):
                #Expects shape [seqlen, num_weather_features, batch, sizes,sizes....]
                seqlen = conv_outs[pastkey].shape[0]
                if len(conv_outs[pastkey].shape) > 3:
                    batchsize = conv_outs[pastkey].shape[2]
                else:
                    batchsize = 1
                features_out = self.convs[0][i].d_model
                conv_outs[curkey] = torch.zeros((seqlen, len(self.features), batchsize, features_out), device=self.device)
                for f in range(len(self.features)):
                    conv_outs[curkey][:,f] = self.convs[f][i](conv_outs[pastkey][:,f])


        # Flatten the convolution layer output
        outkey = 'cnv{}'.format(len(conv_outs)-2)
        fc_out = {'fc-1': conv_outs[outkey].view(-1,self.fc[0].in_features)}
        for i in range(len(self.fc)):
            pastkey = 'fc{}'.format(i-1)
            curkey = 'fc{}'.format(i)
            fc_out[curkey] = F.relu(self.fc[i](fc_out[pastkey]))

        #############################################################
        # input_seq = flight trajectory data + weather features
        # shape: [seq_len, batch_size, input_features]
        fc_outkey = 'fc{}'.format(len(fc_out)-2)
        rnn_input_seq = torch.cat((fc_out[fc_outkey].view(x_t.size(0),x_t.size(1),-1), x_t,), -1)
        rnnout = rnn_input_seq
        for i in range(len(self.rnns)):
            if isinstance(self.rnns[i], indrnn) or isinstance(self.rnns[i], cuda_indrnn):
                rnnlay = 0
                for j in range(len(self.rnns[:i])):
                    if isinstance(self.rnns[j], indrnn) or isinstance(self.rnns[j], cuda_indrnn): rnnlay += 1
                rnnout = self.rnns[i](rnnout, self.hidden_cell[rnnlay])
            elif isinstance(self.rnns[i], BatchNorm) or isinstance(self.rnns[i], torch.nn.Dropout):
                rnnout = self.rnns[i](rnnout)
            elif isinstance(self.rnns[i], torch.nn.BatchNorm1d):
                rnnout = self.rnns[i](rnnout.permute(1,2,0)).permute(2,0,1)
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

    def update_dict(self):
        self.struct_dict = {'class': str(type(self)).split('\'')[1],
                            'device': self.device, 'paradigm': self.paradigm,
                            'conv_input': self.conv_input, 'conv_hidden': self.conv_hidden,
                            'conv_output': self.conv_output, 'attntype': self.attntype,
                            'dense_hidden': self.dense_hidden, 'features': self.features,
                            'rnn_type': self.rnn_type, 'rnn_layers': self.rnn_layers,
                            'rnn_input': self.rnn_input, 'rnn_hidden': self.rnn_hidden,
                            'rnn_output': self.rnn_output, 'hidden_cell': self.hidden_cell, 'droprate': self.droprate,
                            'loss_fn': self.loss_function, 'optim': type(self.optimizer), 'batchnorm': self.batchnorm}
        if hasattr(self, 'sched'):
            self.struct_dict['sched'] = self.sched


    def save_model(self, override_path: str = None, override: bool = False):
        model_name = self.model_name()
        if override_path == None:
            container = 'Models/{}/{}/'.format('&'.join(self.features),model_name)
            model_path = '{}/{}'.format(container, model_name)
            while os.path.isfile(model_path) and not override:
                choice = input("Model Exists:\n1: Replace\n2: New Model\n")
                if choice == '1':
                    break
                elif choice == '2':
                    name = input("Enter model name\n")
                    model_path = '{}/{}'.format(container,name)
            container = '/'.join(model_path.split('/')[:-1])
        else:
            container = override_path
            model_path = '{}/{}'.format(container, model_name)
        if not os.path.isdir(container):
            os.makedirs(container)
        torch.save({'struct_dict': self.struct_dict, 'state_dict': self.state_dict(),
                    'opt_dict': self.optimizer.state_dict(), 'epochs_trained': self.epochs_trained,
                    'batch_size': self.batch_size}, model_path)

    def model_name(self):
        recurrence = str(self.rnn_type).split('\'')[1].split('.')[-1]
        recurrence = recurrence + '{}lay'.format(self.rnn_layers)
        opt = str(self.optimizer.__class__).split('\'')[1].split('.')[-1]
        if self.attntype == 'None':
            convs = 'CONV{}.{}-'.format(len(self.features), self.droprate)
        if self.attntype == 'after':
            convs = 'SAA{}.{}-'.format(len(self.features), self.droprate)
        elif self.attntype == 'replace':
            convs = 'SAR{}.{}-'.format(len(self.features), self.droprate)
        model_name = '{}{}-OPT{}-LOSS{}-EPOCHS{}-BATCH{}-RNN{}_{}_{}'.format(convs, recurrence, opt, self.loss_function,
                                                                                self.epochs_trained,
                                                                                self.batch_size, self.rnn_input,
                                                                                self.rnn_hidden, self.rnn_output)
        return model_name