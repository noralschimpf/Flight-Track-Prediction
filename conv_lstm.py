import torch
import torch.nn as nn
import torch.nn.functional as F

# customized Convolution and LSTM model
class CONV_LSTM(nn.Module):
    def __init__(self, conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16, dense_output=4,\
                 lstm_input=6, lstm_hidden=100, lstm_output=2):
        # conv and lstm input and output parameters can be customized
        super().__init__()
        # convolution layer for weather feature extraction prior to the LSTM
        self.conv_input = conv_input
        self.conv_hidden = conv_hidden
        self.conv_output = conv_output
        self.dense_hidden = dense_hidden
        self.dense_output = dense_output

        self.conv_1 = torch.nn.Conv2d(self.conv_input, self.conv_hidden, kernel_size=6, stride=2)
        self.conv_2 = torch.nn.Conv2d(self.conv_hidden, self.conv_output, kernel_size=3, stride=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 64 input features, 16 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.conv_output*9, self.dense_hidden)
        # 16 input features, 4 output features (see sizing flow below)
        self.fc2 = torch.nn.Linear(self.dense_hidden, self.dense_output)

        ################################################################
        # LSTM model
        # concatenate flight trajectory input with weather features
        self.lstm_input = lstm_input
        self.lstm_hidden = lstm_hidden
        self.lstm_output = lstm_output

        self.lstm = nn.LSTM(self.lstm_input, self.lstm_hidden)
        self.linear = nn.Linear(self.lstm_hidden, self.lstm_output)
        #initiate LSTM hidden cell with 0
        self.hidden_cell = (torch.zeros(1, 1, self.lstm_hidden), torch.zeros(1, 1, self.lstm_hidden))

    def forward(self, x_w, x_t):
        # convolution apply first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)
        x_conv_1 = F.relu(self.conv_1(x_w))
        x_conv_2 = F.relu(self.conv_2(x_conv_1))

        # Flatten the convolution layer output
        x_conv_output = x_conv_2.view(x_conv_2.size(0), -1)
        # print('x_conv_output size:', x_conv_output.size())
        x_fc_1 = F.relu(self.fc1(x_conv_output))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 256) to (1, 64)
        x_fc_2 = F.relu(self.fc2(x_fc_1))

        #############################################################
        # input_seq = flight trajectory data + weather features
        lstm_input_seq = torch.cat((x_fc_2.detach().view(-1, 4), x_t.view(-1, 2)), 1)

        # feed input_seq into LSTM model
        lstm_out, self.hidden_cell = self.lstm(lstm_input_seq.view(len(lstm_input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(lstm_input_seq), -1))
        # print(predictions)
        return predictions
