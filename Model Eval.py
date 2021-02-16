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
dev = 'cuda:1'

# open model and data
mdl = load_model('Models/CONV-LSTM-OPTAdam-LOSSMSELoss()-EPOCHS20-LSTM6_100_2')
df_test = pd.read_csv('test_flight_samples.txt')
fp_test, ft_test, wc_test = df_test['flight plans'].to_list(), df_test['flight tracks'].to_list(), df_test['weather cubes'].to_list()
flight_data = CustomDataset(root_dir=root, abspath_fp=fp_test, abspath_ft=ft_test, abspath_wc=wc_test, transform=ToTensor(), device='cpu')
test_flights = torch.utils.data.DataLoader(flight_data, collate_fn=None, batch_size=1, num_workers=8, pin_memory=True, shuffle=False, drop_last=True)

# begin validation
flight_losses = torch.zeros(len(test_flights))
for idx, (fp, ft, wc) in tqdm.tqdm(enumerate(test_flights), desc='eval flights', leave=False, position=0):
    fp = fp[:, :, 1:].cuda(device=mdl.device, non_blocking=True)
    ft = ft[:, :, 1:].cuda(device=mdl.device, non_blocking=True)
    wc = wc.cuda(device=mdl.device, non_blocking=True)

    mdl.optimizer.zero_grad()
    mdl.hidden_cell = (
        torch.repeat_interleave(fp[0,0,0], mdl.lstm_hidden).view(1, 1, mdl.lstm_hidden),
        torch.repeat_interleave(fp[0,0,1], mdl.lstm_hidden).view(1, 1, mdl.lstm_hidden))
    y_pred = mdl(wc.reshape((-1, 1, 20, 20)), fp[:])
    flight_losses[idx] = mdl.loss_function(y_pred, ft[:].view(-1, 2))

    if mdl.device.__contains__('cuda'):
        y_pred, fp, ft = y_pred.cpu(), fp.cpu(), ft.cpu()
    y_pred, fp, ft = y_pred.detach().numpy(), fp.detach().numpy(), ft.detach().numpy()

    df_flight = pd.DataFrame(data={'flight plan LAT': fp[0,:, 0], 'flight plan LON': fp[0,:, 1],
                                   'prediction LAT': y_pred[:, 0], 'prediction LON': y_pred[:, 1],
                                   'actual LAT': ft[0,:, 0], 'actual LON': ft[0,:, 1]})
    df_flight.to_csv('Output/eval {}'.format(flight_data.get_flightname(idx)))

if mdl.device.__contains__('cuda'):
    flight_losses = flight_losses.cpu()
flight_losses = flight_losses.detach().numpy()

flight_names = [flight_data.get_flightname(x) for x in range(len(flight_losses))]
df_losses = pd.DataFrame(data={'flight name': flight_names, 'loss (MSE)': flight_losses})
df_losses.to_csv('Output/flight losses')
