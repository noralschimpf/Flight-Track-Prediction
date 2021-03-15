import pandas as pd
import torch
import tqdm
import os
from model import load_model, CONV_LSTM, CONV_GRU
from custom_dataset import CustomDataset, ToTensor, pad_batch


def main():
    torch.multiprocessing.set_start_method('spawn')
    # Load model and specified set of test flights
    # For each Model
    #       Collect Predicted and Actual Trajectory into flight-specific CSV
    #       Store loss (MSE, rmse,mae, mape) in general evaluation CSV

    root = 'data/'
    output = 'Output/'
    dev = 'cuda:1'

    # open model and data
    mdl_paths = [os.path.join(os.path.abspath('.'),'Models/{}/{}'.format(x,x)) for x in os.listdir('Models') if os.path.isdir('Models/{}'.format(x))]
    mdl_dirs = ['/'.join(x.split('/')[:-1]) for x in mdl_paths]
    mdls = []

    for path in mdl_paths:
        mdls.append(load_model(path))

    df_test = pd.read_csv('test_flight_samples.txt')
    fp_test, ft_test, wc_test = df_test['flight plans'].to_list(), df_test['flight tracks'].to_list(), df_test['weather cubes'].to_list()
    flight_data = CustomDataset(root_dir=root, abspath_fp=fp_test, abspath_ft=ft_test, abspath_wc=wc_test,
                                transform=ToTensor(), device='cpu')
    test_flights = torch.utils.data.DataLoader(flight_data, collate_fn=pad_batch, batch_size=1, num_workers=8,
                                               pin_memory=True, shuffle=False, drop_last=True)

    # Generate MinMaxScalers

    # begin validation
    for i in range(len(mdls)):
        mdlname = mdls[i].model_name(epochs=500, batch_size=1)
        if not os.path.isdir('Output/{}'.format(mdlname)):
            os.mkdir('Output/{}'.format(mdlname))

        print(mdls[i])

        mdls[i].eval()
        with torch.no_grad():
            flight_losses = torch.zeros(len(test_flights))
            for idx, (fp, ft, wc) in enumerate(tqdm.tqdm(test_flights, desc='eval flights', leave=False, position=0)):
                if mdls[i].device.__contains__('cuda'):
                    fp = fp[:, :, 1:].cuda(device=mdls[i].device, non_blocking=True)
                    ft = ft[:, :, 1:].cuda(device=mdls[i].device, non_blocking=True)
                    wc = wc.cuda(device=mdls[i].device, non_blocking=True)
                else:
                    fp, ft = fp[:,:,1:], ft[:,:,1:]

                mdls[i].optimizer.zero_grad()
                #TODO： T/except block for hidden cells
                if isinstance(mdls[i], CONV_LSTM):
                    mdls[i].hidden_cell = (
                        torch.repeat_interleave(fp[0,0,0], mdls[i].lstm_hidden).view(1, 1, mdls[i].lstm_hidden),
                        torch.repeat_interleave(fp[0,0,1], mdls[i].lstm_hidden).view(1, 1, mdls[i].lstm_hidden))
                elif isinstance(mdls[i], CONV_GRU):
                    mdls[i].hidden_cell = torch.hstack((
                        torch.repeat_interleave(fp[:, 0, 0], int(mdls[i].gru_hidden / 2)),
                        torch.repeat_interleave(fp[:, 0, 1], int(mdls[i].gru_hidden / 2))
                        )).view(1, -1, int(mdls[i].gru_hidden))

                y_pred = mdls[i](wc, fp)
                flight_losses[idx] = mdls[i].loss_function(y_pred, ft)

                if mdls[i].device.__contains__('cuda'):
                    y_pred, fp, ft = y_pred.cpu(), fp.cpu(), ft.cpu()
                y_pred, fp, ft = y_pred.detach().numpy(), fp.detach().numpy(), ft.detach().numpy()

                df_flight = pd.DataFrame(data={'flight plan LAT': fp[0,:, 0], 'flight plan LON': fp[0,:, 1],
                                               'prediction LAT': y_pred[0, :, 0], 'prediction LON': y_pred[0, :, 1],
                                               'actual LAT': ft[0,:, 0], 'actual LON': ft[0,:, 1]})
                df_flight.to_csv('Output/{}/eval {}'.format(mdlname, flight_data.get_flightname(idx)))

            if mdls[i].device.__contains__('cuda'):
                flight_losses = flight_losses.cpu()
            flight_losses = flight_losses.detach().numpy()

            flight_names = [flight_data.get_flightname(x) for x in range(len(flight_losses))]
            df_losses = pd.DataFrame(data={'flight name': flight_names, 'loss (MSE)': flight_losses})
            df_losses.to_csv('Models/{}/flight losses.txt'.format(mdlname))

if __name__ == '__main__':
    main()