import pandas as pd
import torch
import tqdm
import os
from libmodels import model
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from custom_dataset import CustomDataset, ToTensor, pad_batch, ValidFiles
from global_vars import flight_mins, flight_min_tol
import logging
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main():
    logging.basicConfig(filemode='w', filename='Evaluation.log')
    scale = True
    torch.multiprocessing.set_start_method('spawn')
    # Load model and specified set of test flights
    # For each Model
    #       Collect Predicted and Actual Trajectory into flight-specific CSV
    #       Store loss (MSE, rmse,mae, mape) in general evaluation CSV
    root_testdir = '/home/dualboot/Data/Test'
    root_valdir = '/home/dualboot/Data/Train'
    output = 'Output/'
    dev = 'cpu'

    # Collect model groups
    valid_products = ['ECHO_TOP','VIL','tmp','uwind','vwind']
    mdl_product_dirs = [os.path.join(os.path.abspath('.'), 'Models/{}'.format(x)) for x in os.listdir('Models') if
                        os.path.isdir('Models/{}'.format(x)) and any([y in x for y in valid_products])]
    mdl_dirs = [os.path.join(x,y) for x in mdl_product_dirs for y in os.listdir(x) if 'EPOCHS' in y][:4]
    [print(x) for x in mdl_dirs]
    mdls = []

    # Generate Test Data
    list_products = ['ECHO_TOP']; cube_height = 1
    #flight_mins = {'KJFK_KLAX': 5*60, 'KIAH_KBOS': 3.5*60, 'KATL_KORD': 1.5*60,
     #              'KATL_KMCO': 1.25*60, 'KSEA_KDEN': 2.25*60, 'KORD_KLGA': 1.5}

    #IF TRAINED 1ROUTE EVAL LARGER SET
    fps_val, fts_val, wcs_val, dates_val, _ = ValidFiles(root_valdir, list_products, under_min=flight_min_tol,
                                                        fp_subdir='/Flight Plans/Sorted-interp',
                                                        ft_subdir='/Flight Tracks/Interpolated')

    fps_test, fts_test, wcs_test, dates, _ = ValidFiles(root_testdir, list_products, under_min=flight_min_tol,
                     fp_subdir='/Flight Plans/Sorted-interp', ft_subdir='/Flight Tracks/Interpolated')
    test_dataset = CustomDataset(root_testdir, fps_test, fts_test, wcs_test, list_products, ToTensor(), device='cpu')
    test_dl = torch.utils.data.DataLoader(test_dataset, collate_fn=pad_batch, batch_size=1, num_workers=0, pin_memory=True,
                         shuffle=False, drop_last=True)
    df_test = pd.DataFrame(data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
    df_test.to_csv('Test Dataset Valid Flights.csv')

    for mdl_group in tqdm.tqdm(mdl_dirs, desc='model', leave=False, position=0):
        mdl_splits = [x for x in os.listdir(mdl_group) if 'fold' in x and not 'IGNORE' in x and not 'EPOCHS0' in x and not x=='fold0']
        if len(mdl_splits) > 0:
            df_total_losses = pd.DataFrame(columns=['flight name', 'loss (MSE)'])
            for fold in tqdm.tqdm(mdl_splits, desc='fold', leave=False, position=1):
                foldpath = os.path.join(mdl_group, fold)
                mdlname = [x for x in os.listdir(foldpath) if 'EPOCHS' in x][0]
                mdl = model.load_model(os.path.join(mdl_group, fold, mdlname))
                df_val = pd.read_csv('{}/test_flight_samples.txt'.format(foldpath))
                fp_test, ft_test, wc_test = df_val['flight plans'].to_list(), df_val['flight tracks'].to_list(), df_val['weather cubes'].to_list()
                trainroutes = [x.split('/')[-1][:9] for x in fp_test]
                fp_test += [x for x in fps_val if not x.split('/')[-1][:9] in trainroutes]
                ft_test += [x for x in fts_val if not x.split('/')[-1][:9] in trainroutes]
                wc_test += [f"[\'{x[0]}\']" for x in wcs_val if not x[0].split('/')[-1][:9] in trainroutes]
                del df_val; df_val = pd.DataFrame()
                df_val['flight plans'], df_val['flight tracks'], df_val['weather cubes'] = fp_test, ft_test, wc_test
                wc_test = [[a.split('\'')[1]] for a in wc_test]

                '''
                # ONLY IF TRAINING PERFORMED ON SEPARATE MACHINE
                for pathlist in [fp_test, ft_test, wc_test]:
                    for i in range(len(pathlist)):
                        pathlist[i] = pathlist[i].replace(pathlist[i][:pathlist[i].index('data')],os.path.abspath('.') + '/')
                '''
                # Generate Validation Data
                val_dataset = CustomDataset(root_dir=root_valdir, abspath_fp=fp_test, abspath_ft=ft_test, list_products=mdl.features, list_abspath_wc=wc_test,
                                        transform=ToTensor(), device='cpu')
                val_dl = torch.utils.data.DataLoader(val_dataset, collate_fn=pad_batch, batch_size=1, num_workers=0,
                                                       pin_memory=True, shuffle=False, drop_last=True)

                    # begin validation
                #mdlname = mdl.model_name()
                mdlname = mdl_group.split('/')[-1]
                prdstr = '&'.join(mdl.features)
                if not os.path.isdir('Output/{}/{}'.format(prdstr,mdlname)):
                    os.makedirs('Output/{}/{}'.format(prdstr,mdlname))

                mdl.eval()
                with torch.no_grad():
                    for dl in tqdm.tqdm([val_dl, test_dl], desc='Dataset', leave=False, position=2):
                        if dl == val_dl:
                            df = df_val
                            dset = val_dataset
                        elif dl == test_dl:
                            df = df_test
                            dset = test_dataset
                        flight_losses = torch.zeros(len(dl), device=mdl.device)
                        try:
                            for idx, (fp, ft, wc) in enumerate(tqdm.tqdm(dl, desc='flight', leave=False, position=3)):

                                if mdl.device.__contains__('cuda'):
                                    fp = fp[:, :, :].cuda(device=mdl.device, non_blocking=True)
                                    ft = ft[:, :, :].cuda(device=mdl.device, non_blocking=True)
                                    wc = wc.cuda(device=mdl.device, non_blocking=True)
                                else:
                                    fp, ft = fp[:,:,:], ft[:,:,:]
                                if scale:
                                    # scale lats 24 - 50 -> 0-1
                                    fp[:, :, 0] = (fp[:, :, 0] - 24.) / (50. - 24.)
                                    ft[:, :, 0] = (ft[:, :, 0] - 24.) / (50. - 24.)

                                    # scale lons -126 - -66-> 0-1
                                    fp[:, :, 1] = (fp[:, :, 1] + 126.) / (-66. + 126.)
                                    ft[:, :, 1] = (ft[:, :, 1] + 126.) / (-66. + 126.)

                                    # scale alts/ETs -1000 - 64000 -> 0-1
                                    fp[:, :, 2] = (fp[:, :, 2] + 1000.) / (64000. + 1000.)
                                    ft[:, :, 2] = (ft[:, :, 2] + 1000.) / (64000. + 10000)
                                    wc = (wc + 1000.) / (64000. + 1000.)

                                mdl.optimizer.zero_grad()

                                lat, lon, alt = fp[0, :, 0], fp[0, :, 1], fp[0, :, 2]
                                coordlen = int(mdl.rnn_hidden / 3)
                                padlen = mdl.rnn_hidden - 3 * coordlen
                                tns_coords = torch.vstack((lat.repeat(coordlen).view(-1, val_dl.batch_size),
                                                           lon.repeat(coordlen).view(-1, val_dl.batch_size),
                                                           alt.repeat(coordlen).view(-1, val_dl.batch_size),
                                                           torch.zeros(padlen, len(lat),
                                                                   device=mdl.device))).T.view(1, -1, mdl.rnn_hidden)
                                mdl.init_hidden_cell(tns_coords)

                                y_pred = mdl(wc, fp)
                                flight_losses[idx] = mdl.loss_function(y_pred, ft)

                                if mdl.device.__contains__('cuda'):
                                    y_pred, fp, ft = y_pred.cpu(), fp.cpu(), ft.cpu()
                                y_pred, fp, ft = y_pred.detach().numpy(), fp.detach().numpy(), ft.detach().numpy()



                                # Scale Outputs
                                lat_scaler = MinMaxScaler([0,1]); lat_scaler.fit(np.array([[24.], [50.]]))
                                lon_scaler = MinMaxScaler([0,1]); lon_scaler.fit(np.array([[-126.], [-66.]]))
                                alt_scaler = MinMaxScaler([0,1]);alt_scaler.fit(np.array([[-1000.], [64000.]]))

                                fp[:,0,0] = lat_scaler.inverse_transform(fp[:,0,0].reshape(-1,1)).reshape(-1)
                                fp[:,0,1] = lon_scaler.inverse_transform(fp[:,0,1].reshape(-1,1)).reshape(-1)
                                fp[:,0,2] = alt_scaler.inverse_transform(fp[:,0,2].reshape(-1,1)).reshape(-1)
                                y_pred[:, 0, 0] = lat_scaler.inverse_transform(y_pred[:, 0, 0].reshape(-1,1)).reshape(-1)
                                y_pred[:, 0, 1] = lon_scaler.inverse_transform(y_pred[:, 0, 1].reshape(-1,1)).reshape(-1)
                                y_pred[:, 0, 2] = alt_scaler.inverse_transform(y_pred[:, 0, 2].reshape(-1,1)).reshape(-1)
                                ft[:, 0, 0] = lat_scaler.inverse_transform(ft[:, 0, 0].reshape(-1,1)).reshape(-1)
                                ft[:, 0, 1] = lon_scaler.inverse_transform(ft[:, 0, 1].reshape(-1,1)).reshape(-1)
                                ft[:, 0, 2] = alt_scaler.inverse_transform(ft[:, 0, 2].reshape(-1,1)).reshape(-1)

                                if not os.path.isdir(f'Output/{prdstr}/{mdlname}/Denormed'):
                                    os.makedirs(f'Output/{prdstr}/{mdlname}/Denormed')

                                df_flight = pd.DataFrame(data=
                                                 {'flight plan LAT': fp[:,0, 0], 'flight plan LON': fp[:,0, 1], 'flight plan ALT': fp[:,0,2],
                                                   'predicted LAT': y_pred[:,0, 0], 'predicted LON': y_pred[:,0, 1], 'predicted ALT': y_pred[:,0,2],
                                                   'actual LAT': ft[:,0, 0], 'actual LON': ft[:,0, 1], 'actual ALT': ft[:,0,2]})
                                df_flight.to_csv('Output/{}/{}/Denormed/{} {}'.format(prdstr, mdlname, 'eval' if dl == val_dl else 'test{}'.format(fold[-3:]),
                                                                             '_'.join(df['flight plans'][idx].split('/')[-2:])))
                        except FileNotFoundError as e:
                            logging.warning(e)

                        if mdl.device.__contains__('cuda'):
                            flight_losses = flight_losses.cpu()
                        flight_losses = flight_losses.detach().numpy()

                        flight_names = [dset.get_flightname(x) for x in range(len(flight_losses))]
                        df_losses = pd.DataFrame(data={'flight name': flight_names, 'loss (MSE)': flight_losses})
                        df_total_losses = df_total_losses.append(df_losses)
                        df_losses.to_csv('Models/{}/{}/{}/flight {} losses.txt'.format(prdstr, mdlname,fold, 'eval' if dl == val_dl else 'test'))

                del mdl

            df_total_losses.to_csv('Models/{}/{}/total flight losses.txt'.format(prdstr, mdlname))

if __name__ == '__main__':
    main()
