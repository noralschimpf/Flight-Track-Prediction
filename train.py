import torch
import os, shutil, gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import tqdm
from datetime import datetime
from custom_dataset import CustomDataset, ValidFiles, SplitStrList, pad_batch
from custom_dataset import ToTensor
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
#from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from torch.utils.data import DataLoader
from fit import fit
from ray import tune
from Utils.Misc import str_to_list


'''TRAINING CONFIGS
3D EXPECTED: 1000 Epochs, Batch Size 64 -> MSE .001 Train/Test 90/10
3D EXPECTED: 500 Epochs -> MSE .001 (no batching)   Train/Test 75/25
4D EXPECTED: 500 Epochs -> MSE .03                  Train/Test 75/25

'''


def main():
    warnings.filterwarnings('ignore')
    if os.name == 'nt':
        torch.multiprocessing.set_start_method('spawn')

    # training params
    epochs = 300
    bs = 1
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}
    folds = 4

    dev = 'cuda:1'
    #dev = 'cpu'
    # root_dir = '/media/lab/Local Libraries/TorchDir'
    root_dir = 'data/'  # TEST DATA
    # root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'


    ## MODEL PARAMETERS
    #atts = ['None','after', 'replace']
    atts = ['None']
    #recur_types = [torch.nn.LSTM, torch.nn.GRU, indrnn]
    recur_types = [torch.nn.LSTM]
    rnn_lays = [1]
    drop = 0.0



    # Uncomment block if generating valid file & split files
    total_products=['ECHO_TOP','VIL','uwind','vwind','tmp']
    #list_products=[['ECHO_TOP'], ['VIL'],['tmp'],['vwind'],['uwind']]
    list_products = [['ECHO_TOP']]
    fps, fts, wcs, dates, _ = ValidFiles(root_dir, total_products, under_min=100)
    total_flights = len(fps)

    cnnlstm = tune.Analysis('~/ray_results/CNN_LSTM')
    cnngru = tune.Analysis('~/ray_results/CNN_GRU')
    cnnindrnn = tune.Analysis('~/ray_results/CNN_IndRNN')
    saalstm = tune.Analysis('~/ray_results/CNN+SA_LSTM')
    sarlstm = tune.Analysis('~/ray_results/SA_LSTM')
    cfgs = [x.get_best_config(metric='valloss',mode='min') for x in [cnngru, cnnlstm, saalstm,sarlstm]]
    cfg_lstm = cnnlstm.get_best_config(metric='valloss', mode='min')
    # Correct Models
    for config in cfgs:
        config['epochs'] = epochs
        config['device'] = dev
        if isinstance(config['optim'], str):
            if 'Adam' in config['optim']:
                config['optim'] = torch.optim.Adam
            elif 'RMS' in config['optim']:
                config['optim'] = torch.optim.RMSprop
            elif 'SGD' in config['optim']:
                config['optim'] = torch.optim.SGD
        if isinstance(config['rnn'], str):
            if 'LSTM' in config['rnn']:
                config['rnn'] = torch.nn.LSTM
            elif 'GRU' in config['rnn']:
                config['rnn'] = torch.nn.GRU
            elif 'IndRNN' in config['rnn']:
                config['rnn'] = indrnn
        if isinstance(config['HLs'], str): config['HLs'] = str_to_list(config['HLs'], int)
        if isinstance(config['ConvCh'], str): config['ConvCh'] = str_to_list(config['ConvCh'], int)
        for key in config:
            if 'RNN' in key: config[key] = cfg_lstm[key]

    for products in list_products:
        cube_height = 3 if 'uwind' in products or 'vwind' in products or 'tmp' in products else 1
        prdstr = '&'.join(products)
        for fold in range(folds):
            # Random split
            #train_flights = np.random.choice(total_flights, int(total_flights * .75), replace=False)
            #test_flights = list(set(range(len(fps))) - set(train_flights))

            #cross-validation split
            foldstr = 'fold{}-{}'.format(fold+1,folds)
            test_flights = list(range( int(fold*total_flights/folds), int(((fold+1)*total_flights)/folds) ))
            train_flights = list(set(range(total_flights)) - set(test_flights))
            print('fold {}/{}\t{}-{} test flights'.format(fold+1,folds,min(test_flights),max(test_flights)))
            train_flights.sort()
            test_flights.sort()

            fps_train, fps_test = SplitStrList(fps, test_flights)
            fts_train, fts_test = SplitStrList(fts, test_flights)
            wcs_train, wcs_test = SplitStrList(wcs, test_flights)

            df_trainfiles = pd.DataFrame(
                data={'flight plans': fps_train, 'flight tracks': fts_train, 'weather cubes': wcs_train})
            df_testfiles = pd.DataFrame(data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
            df_trainfiles.to_csv('train_flight_samples.txt'.format(foldstr))
            df_testfiles.to_csv('test_flight_samples.txt'.format(foldstr))


            '''
            # Uncomment block if validated & split files already exist
            df_trainfiles = pd.read_csv('train_flight_samples.txt')
            print('Loading Train Files')
            fps_train, fts_train, wcs_train, train_flights = [],[],[],[]
            fps_train, fts_train = df_trainfiles['flight plans'].values, df_trainfiles['flight tracks'].values
            wcs_train, train_flights = df_trainfiles['weather cubes'].values, list(range(len(fps_train)))
            #TODO: eliminate train_flights    
            '''

            train_dataset = CustomDataset(root_dir, fps_train, fts_train, wcs_train, products, ToTensor(), device='cpu')
            test_dataset = CustomDataset(root_dir, fps_test, fts_test, wcs_test, products, ToTensor(), device='cpu')

            # train_model
            for config in cfgs:
                mdl = fit(config, train_dataset, test_dataset, raytune=False)
                mdl.epochs_trained = config['epochs']
                mdl.save_model(override=True)
                shutil.rmtree('Models/{}/{}'.format(prdstr, mdl.model_name().replace('EPOCHS{}'.format(mdl.epochs_trained), 'EPOCHS0')))
                #os.makedirs('Models/{}/{}'.format(mdl.model_name(), foldstr))
                fold_subdir = os.path.join('Models',prdstr, mdl.model_name(), foldstr)
                if not os.path.isdir(fold_subdir):
                    os.makedirs(fold_subdir)
                for fname in os.listdir('Models/{}/{}'.format(prdstr,mdl.model_name())):
                    if os.path.isfile(os.path.join('Models',prdstr,mdl.model_name(), fname)):
                        shutil.move(os.path.join('Models',prdstr,mdl.model_name(), fname),
                                    os.path.join('Models',prdstr, mdl.model_name(),foldstr, fname))

                #shutil.move('Models/{}'.format(mdl.model_name()), 'Models/{}/{}'.format(mdl.model_name(), foldstr))
                edtime = datetime.now()
                #print('DONE: {}'.format(edtime - sttime))

                if not os.path.isdir('Initialized Plots/{}/{}/{}'.format(prdstr, mdl.model_name(), foldstr)):
                    os.makedirs('Initialized Plots/{}/{}/{}'.format(prdstr, mdl.model_name(), foldstr))
                if not os.path.isdir('Models/{}/{}/{}'.format(prdstr, mdl.model_name(), foldstr)):
                    os.makedirs('Models/{}/{}/{}'.format(prdstr, mdl.model_name(), foldstr))

                plots_to_move = [x for x in os.listdir('Initialized Plots') if x.__contains__('.png')]
                for plot in plots_to_move:
                    shutil.copy('Initialized Plots/{}'.format(plot), 'Initialized Plots/{}/{}/{}/{}'.format(prdstr, mdl.model_name(),foldstr, plot))
                    shutil.copy('Initialized Plots/{}'.format(plot), 'Initialized Plots/{}/{}/{}/{}'.format(prdstr, mdl.model_name(),foldstr, plot))
                    os.remove('Initialized Plots/{}'.format(plot))

                shutil.copy('test_flight_samples.txt', 'Models/{}/{}/{}/test_flight_samples.txt'.format(prdstr, mdl.model_name(), foldstr))
                shutil.copy('train_flight_samples.txt', 'Models/{}/{}/{}/train_flight_samples.txt'.format(prdstr, mdl.model_name(), foldstr))
                shutil.copy('model_epoch_losses.txt', 'Models/{}/{}/{}/model_epoch_losses.txt'.format(prdstr, mdl.model_name(), foldstr))


'''def fit(mdl: CONV_RECURRENT, train_dl: torch.utils.data.DataLoader, test_dl: torch.utils.data.DataLoader, epochs: int, model_name: str = 'Default', ):
    epoch_losses = torch.zeros(epochs, device=mdl.device)
    epoch_test_losses = torch.zeros(epochs, device=mdl.device)
    for ep in tqdm.trange(epochs, desc='{} epoch'.format(mdl.model_name().replace('-OPTAdam','').replace('LOSS','')), position=0, leave=False):
        losses = torch.zeros(len(train_dl), device=mdl.device)

        for batch_idx, (fp, ft, wc) in enumerate(
                tqdm.tqdm(train_dl, desc='flight', position=1, leave=False)):  # was len(flight_data)
            # Extract flight plan, flight track, and weather cubes
            if mdl.device.__contains__('cuda'):
                fp = fp[:, :, :].cuda(device=mdl.device, non_blocking=True)
                ft = ft[:, :, :].cuda(device=mdl.device, non_blocking=True)
                wc = wc.cuda(device=mdl.device, non_blocking=True)
            else:
                fp, ft = fp[:, :, :], ft[:, :, :]
            if mdl.paradigm == 'Regression':
                print("\nFlight {}/{}: ".format(batch_idx + 1, len(train_dl)) + str(len(fp)) + " points")
                for pt in tqdm.trange(len(wc)):
                    mdl.optimizer.zero_grad()
                    lat, lon, alt = fp[0][0].clone().detach(), fp[0][1].clone().detach(). fp[0][2].clone().detach()
                    if mdl.rnn_type == torch.nn.LSTM:
                        mdl.hidden_cell = (
                            lat.repeat(1, 1, mdl.rnn_hidden),
                            lon.repeat(1, 1, mdl.rnn_hidden),
                            alt.repeate(1,1,mdl.rnn_hidden))
                    elif mdl.rnn_type == torch.nn.GRU:
                        mdl.hidden_cell = torch.cat(lat.repeat(1, 1, mdl.rnn_hidden / 3),
                                                    lon.repeat(1, 1, mdl.rnn_hidden / 3),
                                                    alt.repeat(1, 1, mdl.rnn_hidden / 3),
                                                    torch.zeros(1, 1, mdl.rnn_hidden - (3*int(mdl.rnn_hidden/3))))
                    y_pred = mdl(wc[:pt + 1], fp[:pt + 1])
                    # print(y_pred)
                    single_loss = mdl.loss_function(y_pred, ft[:pt + 1].view(-1, 2))
                    if batch_idx < len(train_dl) - 1 and pt % 50 == 0:
                        single_loss.backward()
                        mdl.optimizer.step()
                    if batch_idx == len(train_dl) - 1:
                        losses = torch.cat((losses, single_loss.view(-1)))
            elif mdl.paradigm == 'Seq2Seq':
                mdl.optimizer.zero_grad()

                lat, lon, alt = fp[0,:,0], fp[0,:,1], fp[0,:,2]
                coordlen = int(mdl.rnn_hidden/3)
                padlen = mdl.rnn_hidden - 3*coordlen
                tns_coords = torch.vstack((lat.repeat(coordlen).view(-1, train_dl.batch_size),
                                        lon.repeat(coordlen).view(-1, train_dl.batch_size),
                                        alt.repeat(coordlen).view(-1, train_dl.batch_size),
                                        torch.zeros(padlen, len(lat),
                                        device=mdl.device))).T.view(1,-1,mdl.rnn_hidden)
                mdl.init_hidden_cell(tns_coords)

                y_pred = mdl(wc, fp[:])
                single_loss = mdl.loss_function(y_pred, ft)
                losses[batch_idx] = single_loss.view(-1).detach().item()
                single_loss.backward()
                mdl.optimizer.step()

            if batch_idx == len(train_dl) - 1:
                epoch_losses[ep] = torch.mean(losses).view(-1)

        # TODO: ADD EVAL ON TEST DATA
        mdl.eval()
        test_losses = torch.zeros(len(test_dl), device=mdl.device)
        for test_batch, (fp, ft, wc) in enumerate(test_dl):
            if mdl.device.__contains__('cuda'):
                fp = fp[:, :, :].cuda(device=mdl.device, non_blocking=True)
                ft = ft[:, :, :].cuda(device=mdl.device, non_blocking=True)
                wc = wc.cuda(device=mdl.device, non_blocking=True)
            else:
                fp, ft = fp[:, :, :], ft[:, :, :]

            if mdl.paradigm == 'Seq2Seq':
                mdl.optimizer.zero_grad()
                lat, lon, alt = fp[0, :, 0], fp[0, :, 1], fp[0, :, 2]
                coordlen = int(mdl.rnn_hidden / 3)
                padlen = mdl.rnn_hidden - 3 * coordlen
                tns_coords = torch.vstack((lat.repeat(coordlen).view(-1, test_dl.batch_size),
                                           lon.repeat(coordlen).view(-1, test_dl.batch_size),
                                           alt.repeat(coordlen).view(-1, test_dl.batch_size),
                                           torch.zeros(padlen, len(lat),
                                                       device=mdl.device))).T.view(1, -1, mdl.rnn_hidden)
                mdl.init_hidden_cell(tns_coords)

                y_pred = mdl(wc, fp[:])
                single_loss = mdl.loss_function(y_pred, ft)
                test_losses[test_batch] = single_loss.view(-1).detach().item()
        epoch_test_losses[ep] = test_losses.mean().view(-1)
        mdl.train()

        if ep % 10 == 0:
            if mdl.device.__contains__('cuda'):
                losses = losses.cpu()
                test_losses = test_losses.cpu()
            plt.plot(losses.detach().numpy(), label='train data')
            plt.plot(test_losses.detach().numpy(), label='test data')
            plt.legend()
            plt.title('Losses (Epoch {})'.format(ep + 1))
            plt.xlabel('Flight')
            plt.ylabel('Loss (MSE)')
            # plt.savefig('Eval Epoch{}.png'.format(ep+1), dpi=400)
            plt.savefig('Initialized Plots/Eval Epoch{}.png'.format(ep + 1), dpi=400)
            plt.close()
            del losses
            gc.collect()
        if ep % 50 == 0:
            mdl.save_model(override=True)



    if mdl.device.__contains__('cuda'):
        epoch_losses = epoch_losses.cpu()
        epoch_test_losses = epoch_test_losses.cpu()
    e_losses = epoch_losses.detach().numpy()
    e_test_losses = epoch_test_losses.detach().numpy()

    plt.plot(e_losses, label='train data')
    plt.plot(e_test_losses, label='test data')
    plt.legend()
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.savefig('Initialized Plots/Model Eval.png', dpi=300)
    plt.close()

    plt.plot(e_losses[:], label='train data')
    plt.plot(e_test_losses[:], label='test data')
    plt.legend()
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.ylim([0, .01])
    plt.yticks(np.linspace(0, .01, 11))
    plt.savefig('Initialized Plots/Model Eval RangeLimit.png', dpi=300)
    plt.close()

    df_eloss = pd.DataFrame({'loss': e_losses})
    df_eloss.to_csv('model_epoch_losses.txt')'''


if __name__ == '__main__':
    main()
