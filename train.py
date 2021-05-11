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
    epochs = 1
    bs = 2
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}
    folds = 4

    dev = 'cuda'
    # dev = 'cpu'
    # root_dir = '/media/lab/Local Libraries/TorchDir'
    root_dir = 'data/'  # TEST DATA
    # root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'



    # Uncomment block if generating valid file & split files
    products=['ECHO_TOP','VIL','uwind','vwind','tmp']
    cube_height = 3 if 'uwind' in products or 'vwind' in products or 'tmp' in products else 1
    fps, fts, wcs, dates, _ = ValidFiles(root_dir, products, under_min=100)
    total_flights = len(fps)

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
        train_dl = DataLoader(train_dataset, collate_fn=pad_batch, batch_size=bs, num_workers=0, pin_memory=True,
                              shuffle=False, drop_last=True)

        test_dataset = CustomDataset(root_dir, fps_test, fts_test, wcs_test, products, ToTensor(), device='cpu')
        test_dl = DataLoader(test_dataset, collate_fn=pad_batch, batch_size=1, num_workers=0, pin_memory=True,
                              shuffle=False, drop_last=True)

        # train_model
        for recur in [torch.nn.LSTM, torch.nn.GRU, indrnn]:
        #for recur in [torch.nn.LSTM]:
            for rnn_lay in [1,2]:
            #for rnn_lay in [2]:
                for att in ['None','after', 'replace']:
                #for att in ['replace']:
                    rlay = rnn_lay
                    if recur == indrnn or recur == cuda_indrnn: rlay += 1
                    mdl = CONV_RECURRENT(paradigm=paradigms[1], cube_height=cube_height, device=dev, rnn=recur, num_features=len(products),
                                         rnn_layers=rlay, attn=att, batch_size=bs, droprate=.5)
                    mdl.optimizer = torch.optim.Adam(mdl.parameters(), lr=2e-4)
                    print(mdl)
                    sttime = datetime.now()
                    #print('START FIT: {}'.format(sttime))
                    fit(mdl, train_dl, test_dl, epochs, train_flights)
                    mdl.epochs_trained = epochs
                    mdl.save_model(override=True)
                    shutil.rmtree('Models/{}'.format(mdl.model_name().replace('EPOCHS{}'.format(epochs),'EPOCHS0')))
                    #os.makedirs('Models/{}/{}'.format(mdl.model_name(), foldstr))
                    fold_subdir = os.path.join('Models',mdl.model_name(), foldstr)
                    if not os.path.isdir(fold_subdir):
                        os.mkdir(fold_subdir)
                    for fname in os.listdir('Models/{}'.format(mdl.model_name())):
                        if os.path.isfile(os.path.join('Models',mdl.model_name(), fname)):
                            os.rename(os.path.join('Models',mdl.model_name(), fname), os.path.join('Models',mdl.model_name(),foldstr, fname))

                    #shutil.move('Models/{}'.format(mdl.model_name()), 'Models/{}/{}'.format(mdl.model_name(), foldstr))
                    edtime = datetime.now()
                    #print('DONE: {}'.format(edtime - sttime))

                    if not os.path.isdir('Initialized Plots/{}/{}'.format(mdl.model_name(), foldstr)):
                        os.makedirs('Initialized Plots/{}/{}'.format(mdl.model_name(),foldstr))
                    plots_to_move = [x for x in os.listdir('Initialized Plots') if x.__contains__('.png')]
                    for plot in plots_to_move:
                        shutil.move('Initialized Plots/{}'.format(plot), 'Initialized Plots/{}/{}/{}'.format(mdl.model_name(), foldstr, plot))

                    shutil.copy('test_flight_samples.txt', 'Models/{}/{}/test_flight_samples.txt'.format(mdl.model_name(), foldstr))
                    shutil.copy('train_flight_samples.txt', 'Models/{}/{}/train_flight_samples.txt'.format(mdl.model_name(), foldstr))
                    shutil.copy('model_epoch_losses.txt', 'Models/{}/{}/model_epoch_losses.txt'.format(mdl.model_name(), foldstr))


def fit(mdl: CONV_RECURRENT, train_dl: torch.utils.data.DataLoader, test_dl: torch.utils.data.DataLoader, epochs: int, model_name: str = 'Default', ):
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
    df_eloss.to_csv('model_epoch_losses.txt')


if __name__ == '__main__':
    main()
