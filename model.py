import torch, tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# customized Convolution and LSTM model
class CONV_LSTM(nn.Module):
    def __init__(self, paradigm='Seq2Seq', device='cpu', conv_input=1, conv_hidden=2, conv_output=4, dense_hidden=16,
                 dense_output=4, \
                 lstm_input=6, lstm_hidden=100, lstm_output=2):
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
        self.lstm_input = lstm_input
        self.lstm_hidden = lstm_hidden
        self.lstm_output = lstm_output

        self.lstm = nn.LSTM(self.lstm_input, self.lstm_hidden)
        self.linear = nn.Linear(self.lstm_hidden, self.lstm_output)
        # initiate LSTM hidden cell with 0
        self.hidden_cell = (torch.zeros(1, 1, self.lstm_hidden), torch.zeros(1, 1, self.lstm_hidden))

        if self.device.__contains__('cuda'):
            self.cuda(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def forward(self, x_w, x_t):
        # convolution apply first
        # input_seq = flight trajectory data + weather features
        # x_w is flight trajectory data
        # x_t is weather data (time ahead of flight)
        x_conv_1 = F.relu(self.conv_1(x_w))
        x_conv_2 = F.relu(self.conv_2(x_conv_1))

        # Flatten the convolution layer output
        x_conv_output = x_conv_2.view(-1, self.conv_output * 9)
        # print('x_conv_output size:', x_conv_output.size())
        x_fc_1 = F.relu(self.fc1(x_conv_output))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 256) to (1, 64)
        x_fc_2 = F.relu(self.fc2(x_fc_1))

        #############################################################
        # input_seq = flight trajectory data + weather features
        lstm_input_seq = torch.cat((x_fc_2.detach().view(-1, 4), x_t.view(-1, 2)), 1)

        # feed input_seq into LSTM model
        if self.device.__contains__('cuda'):
            lstm_out, self.hidden_cell = self.lstm(lstm_input_seq.view(len(lstm_input_seq), 1, -1).cuda(self.device),
                                                   self.hidden_cell)
        else:
            lstm_out, self.hidden_cell = self.lstm(lstm_input_seq.view(len(lstm_input_seq), 1, -1), self.hidden_cell)

        #TODO: Dimension expansion (eq. 2l in Pang, Xu, Liu)
        predictions = self.linear(lstm_out.view(len(lstm_input_seq), -1))
        # expansion = F.relu(self.fc_exp(predictions))
        # print(predictions)
        return predictions

    def save_model(self,epochs, opt, model_name: str = None):
        if model_name == None:
            model_name = 'CONV-LSTM-OPT{}-LOSS{}-EPOCHS{}-LSTM{}_{}_{}'.format(opt,self.loss_function, epochs, self.lstm_input, self.lstm_hidden, self.lstm_output)
        model_path = 'Models/' + model_name
        while os.path.isfile(model_path):
            choice = input("Model Exists:\n1: Replace\n2: New Model\n")
            if choice == '1':
                break
            elif choice == '2':
                name = input("Enter model name\n")
                model_path = 'Models/' + name
        torch.save(self.state_dict(), model_path)

    def fit(self, flight_data: torch.utils.data.DataLoader, epochs: int, model_name: str = 'Default'):
        # TODO: add graphics: loss over epoch, acc over epoch
        # TODO: BATCH FLIGHTS
        # TODO: Initialize Flight Coords

        epoch_losses = torch.tensor((),device=self.device)
        for ep in range(epochs):
            losses = torch.tensor((),device=self.device)

            rg_flights = None
            if self.paradigm == 'Regression':
                rg_flights = range(len(flight_data))
            elif self.paradigm == 'Seq2Seq':
                rg_flights = tqdm.trange(len(flight_data))

            for i in rg_flights:  # was len(flight_data)
                # Extract flight plan, flight track, and weather cubes
                fp, ft, wc = flight_data[i]
                # TODO: Move to Dataloader
                maxlen = min(len(fp), len(ft), len(wc))
                wc = wc[:maxlen]
                fp = fp[:maxlen, 1:]
                ft = ft[:maxlen, 1:]
                # train_dataset = [wc[i], fp[i], ft[i]]

                if self.paradigm == 'Regression':
                    print("\nFlight {}/{}: ".format(i + 1, len(flight_data)) + str(len(fp)) + " points")
                    for pt in tqdm.trange(len(wc)):
                        self.optimizer.zero_grad()
                        lat, lon = fp[0][0].clone().detach().cuda(self.device), fp[0][1].clone().detach().cuda(self.device)
                        self.hidden_cell = (
                            lat.repeat(1,1,self.lstm_hidden),
                            lon.repeat(1,1,self.lstm_hidden))
                        # TODO: scrub time from files?
                        y_pred = self(wc[:pt + 1].reshape((-1, 1, 20, 20)), fp[:pt + 1])
                        # print(y_pred)
                        single_loss = self.loss_function(y_pred, ft[:pt + 1].view(-1, 2))
                        if i < len(flight_data) - 1 and pt % 50 == 0:
                            single_loss.backward()
                            self.optimizer.step()
                        if i == len(flight_data) - 1:
                            losses = torch.cat((losses, single_loss.view(-1)))
                elif self.paradigm == 'Seq2Seq':
                    self.optimizer.zero_grad()
                    self.hidden_cell = (
                    torch.repeat_interleave(fp[0][0], self.lstm_hidden).view(1, 1, self.lstm_hidden),
                    torch.repeat_interleave(fp[0][1], self.lstm_hidden).view(1, 1, self.lstm_hidden))
                    y_pred = self(wc.reshape((-1, 1, 20, 20)), fp[:])
                    single_loss = self.loss_function(y_pred, ft[:].view(-1, 2))
                    losses = torch.cat((losses, single_loss.view(-1)))
                    single_loss.backward()
                    self.optimizer.step()
            if i == len(flight_data) - 1:
                epoch_losses = torch.cat((epoch_losses, torch.mean(losses).view(-1)))

                if self.device.__contains__('cuda'):
                    losses = losses.cpu()
                plt.plot(losses.detach().numpy())
                plt.title('Training (Epoch {})'.format(ep + 1))
                plt.xlabel('Flight')
                plt.ylabel('Loss (MSE)')
                # plt.savefig('Eval Epoch{}.png'.format(ep+1), dpi=400)
                plt.savefig('Initialized Plots/Eval Epoch{}.png'.format(ep + 1), dpi=400)
                plt.close()
        if self.device.__contains__('cuda'):
            epoch_losses = epoch_losses.cpu()
        plt.plot(epoch_losses.detach().numpy())
        plt.title('Avg Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Loss (MSE)')

    def evaluate(self, flight_data: torch.utils.data.DataLoader):
        for i in range(len(flight_data)):
            fp, ft, wc = flight_data[i]
            wc = wc[:len(fp)]
            print("\nFlight {}/{}: ".format(i + 1, len(flight_data)) + str(len(fp)) + " points")

            losses = torch.zeros(len(wc), requires_grad=False, device=self.device)
            preds = torch.zeros(len(wc), requires_grad=False, device=self.device)
            lbls = torch.zeros(len(wc), requires_grad=False, device=self.device)

            for pt in tqdm.trange(len(wc)):
                test_dataset = [wc[i], fp[i], ft[i]]
                self.optimizer.zero_grad()
                self.hidden_cell(torch.zeros(1, 1, self.lstm_hidden),
                                 torch.zeros(1, 1, self.lstm_hidden))
                y_pred = self(wc[pt].reshape((1, 1, 20, 20)), fp[pt][1:])
                single_loss = self.loss_function(y_pred, ft[i][1:].view(-1, 2)).long().detach().numpy()


# TODO: load methods
def load_model(model_path: str):
    lstm_in, hid_cells, lstm_lay, b, e, tw = model_path.split('/')[-1].split('-')
    lstm_in = int(''.join([x for x in lstm_in if x.isdigit()]))
    hid_cells = int(''.join([x for x in hid_cells if x.isdigit()]))
    lstm_lay = int(''.join([x for x in lstm_lay if x.isdigit()]))
    b = int(''.join([x for x in b if x.isdigit()]))
    e = int(''.join([x for x in e if x.isdigit()]))
    tw = int(''.join([x for x in tw if x.isdigit()]))
    # TODO: READ FROM FILENAME
    state_dict = torch.load(model_path)
    output_size = len(state_dict['output.weight'])

