import pandas as pd
import torch
import tqdm
from model import load_model
from custom_dataset import CustomDataset, ToTensor

# Load model and specified set of test flights
# For each Model
#       Collect Predicted and Actual Trajectory into flight-specific CSV
#       Store loss (MSE, rmse,mae, mape) in general evaluation CSV

root = 'data/'
output = 'Output/'
dev = 'cpu'

# open model and data
mdl = load_model('Models/CONV-LSTM-OPTAdam-LOSSMSELoss()-EPOCHS5-LSTM6_100_2')
test_flights = pd.read_csv('test_flight_samples.txt').values.squeeze()
flight_data = CustomDataset(root_dir=root, transform=ToTensor(), device=dev)
flight_data.validate_sets(under_min=100)

# begin validation
flight_losses = torch.zeros(len(test_flights))
for i in tqdm.trange(len(test_flights)):

    fp, ft, wc = flight_data[test_flights[i]]
    maxlen = min(len(fp), len(ft), len(wc))
    fp, ft, wc = fp[:, 1:], ft[:, 1:], wc[:]

    mdl.optimizer.zero_grad()
    mdl.hidden_cell = (
        torch.repeat_interleave(fp[0][0], mdl.lstm_hidden).view(1, 1, mdl.lstm_hidden),
        torch.repeat_interleave(fp[0][1], mdl.lstm_hidden).view(1, 1, mdl.lstm_hidden))
    y_pred = mdl(wc.reshape((-1, 1, 20, 20)), fp[:])
    flight_losses[i] = mdl.loss_function(y_pred, ft[:].view(-1, 2))

    if mdl.device.__contains__('cuda'):
        y_pred, fp, ft = y_pred.cpu(), fp.cpu(), ft.cpu()
    y_pred, fp, ft = y_pred.detach().numpy(), fp.detach().numpy(), ft.detach().numpy()

    df_flight = pd.DataFrame(data={'flight plan LAT': fp[:, 0], 'flight plan LON': fp[:, 1],
                                   'prediction LAT': y_pred[:, 0], 'prediction LON': y_pred[:, 1],
                                   'actual LAT': ft[:, 0], 'actual LON': ft[:, 1]})
    df_flight.to_csv('Output/eval {}'.format(flight_data.get_flightname(i)))

if mdl.device.__contains__('cuda'):
    flight_losses = flight_losses.cpu()
flight_losses = flight_losses.detach().numpy()

flight_names = [flight_data.get_flightname(x) for x in range(len(flight_losses))]
df_losses = pd.DataFrame(data={'flight name': flight_names, 'loss (MSE)': flight_losses})
df_losses.to_csv('Output/flight losses')
