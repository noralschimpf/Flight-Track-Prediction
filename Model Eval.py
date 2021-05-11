import pandas as pd
import torch
import tqdm
import os
from libmodels import model
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
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

    # Collect model groups
    mdl_dirs = [os.path.join(os.path.abspath('.'),'Models/{}'.format(x)) for x in os.listdir('Models') if os.path.isdir('Models/{}'.format(x)) and 'EPOCHS' in x]
    mdls = []

    for mdl_group in mdl_dirs:
        mdl_splits = [x for x in os.listdir(mdl_group) if 'fold' in x]
        df_total_losses = pd.DataFrame(columns=['flight name', 'loss (MSE)'])
        for fold in mdl_splits:
            foldpath = os.path.join(mdl_group, fold)
            mdlname = [x for x in os.listdir(foldpath) if 'EPOCHS' in x][0]
            mdl = model.load_model(os.path.join(mdl_group, fold, mdlname))
            df_test = pd.read_csv('{}/test_flight_samples.txt'.format(foldpath))
            fp_test, ft_test, wc_test = df_test['flight plans'].to_list(), df_test['flight tracks'].to_list(), df_test['weather cubes'].to_list()

            '''
            # ONLY IF TRAINING PERFORMED ON SEPARATE MACHINE
            for pathlist in [fp_test, ft_test, wc_test]:
                for i in range(len(pathlist)):
                    pathlist[i] = pathlist[i].replace(pathlist[i][:pathlist[i].index('data')],os.path.abspath('.') + '/')
            '''

    flight_data = CustomDataset(root_dir=root, abspath_fp=fp_test, abspath_ft=ft_test, list_abspath_wc=wc_test,
                                transform=ToTensor(), device='cpu')
    test_flights = torch.utils.data.DataLoader(flight_data, collate_fn=pad_batch, batch_size=1, num_workers=0,
                                               pin_memory=True, shuffle=False, drop_last=True)

            # begin validation
            mdlname = mdl.model_name()
            if not os.path.isdir('Output/{}'.format(mdlname)):
                os.mkdir('Output/{}'.format(mdlname))

            mdl.eval()
            print(mdl)
            with torch.no_grad():
                flight_losses = torch.zeros(len(test_flights), device=mdl.device)
                for idx, (fp, ft, wc) in enumerate(tqdm.tqdm(test_flights, desc='eval flights', leave=False, position=0)):
                    if mdl.device.__contains__('cuda'):
                        fp = fp[:, :, :].cuda(device=mdl.device, non_blocking=True)
                        ft = ft[:, :, :].cuda(device=mdl.device, non_blocking=True)
                        wc = wc.cuda(device=mdl.device, non_blocking=True)
                    else:
                        fp, ft = fp[:,:,:], ft[:,:,:]

                    mdl.optimizer.zero_grad()

                    lat, lon, alt = fp[0, :, 0], fp[0, :, 1], fp[0, :, 2]
                    coordlen = int(mdl.rnn_hidden / 3)
                    padlen = mdl.rnn_hidden - 3 * coordlen
                    tns_coords = torch.vstack((lat.repeat(coordlen).view(-1, test_flights.batch_size),
                                               lon.repeat(coordlen).view(-1, test_flights.batch_size),
                                               alt.repeat(coordlen).view(-1, test_flights.batch_size),
                                               torch.zeros(padlen, len(lat),
                                                       device=mdl.device))).T.view(1, -1, mdl.rnn_hidden)
                    mdl.init_hidden_cell(tns_coords)

                    y_pred = mdl(wc, fp)
                    flight_losses[idx] = mdl.loss_function(y_pred, ft)

                    if mdl.device.__contains__('cuda'):
                        y_pred, fp, ft = y_pred.cpu(), fp.cpu(), ft.cpu()
                    y_pred, fp, ft = y_pred.detach().numpy(), fp.detach().numpy(), ft.detach().numpy()

                    df_flight = pd.DataFrame(data=
                                     {'flight plan LAT': fp[:,0, 0], 'flight plan LON': fp[:,0, 1], 'flight plan ALT': fp[:,0,2],
                                       'prediction LAT': y_pred[:,0, 0], 'prediction LON': y_pred[:,0, 1], 'pretion ALT': y_pred[:,0,2],
                                       'actual LAT': ft[:,0, 0], 'actual LON': ft[:,0, 1], 'actual ALT': ft[:,0,2]})
                    df_flight.to_csv('Output/{}/eval {}'.format(mdlname, '_'.join(df_test['flight plans'][idx].split('/')[-2:])))

                if mdl.device.__contains__('cuda'):
                    flight_losses = flight_losses.cpu()
                flight_losses = flight_losses.detach().numpy()

                flight_names = [flight_data.get_flightname(x) for x in range(len(flight_losses))]
                df_losses = pd.DataFrame(data={'flight name': flight_names, 'loss (MSE)': flight_losses})
                df_total_losses = df_total_losses.append(df_losses)
                df_losses.to_csv('Models/{}/{}/flight losses.txt'.format(mdlname,fold))
        df_total_losses.to_csv('Models/{}/total flight losses.txt'.format(mdlname))

if __name__ == '__main__':
    main()