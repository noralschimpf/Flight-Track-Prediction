import torch
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from datetime import datetime
from custom_dataset import CustomDataset, ValidFiles, SplitStrList, pad_batch
from custom_dataset import ToTensor
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent
from torch.utils.data import DataLoader


'''TRAINING CONFIGS
3D EXPECTED: 1000 Epochs, Batch Size 64 -> MSE .001 Train/Test 90/10
3D EXPECTED: 500 Epochs -> MSE .001 (no batching)   Train/Test 75/25
4D EXPECTED: 500 Epochs -> MSE .03                  Train/Test 75/25

'''


def main():
    if os.name == 'nt':
        torch.multiprocessing.set_start_method('spawn')

    # training params
    epochs = 1
    bs = 1
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}

    # dev = 'cuda:1'
    dev = 'cpu'
    # root_dir = '/media/lab/Local Libraries/TorchDir'
    root_dir = 'data/'  # TEST DATA
    # root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'


    # Uncomment block if generating valid file & split files
    fps, fts, wcs, dates, _ = ValidFiles(root_dir, under_min=100)
    total_flights = len(fps)
    train_flights = np.random.choice(total_flights, int(total_flights * .75), replace=False)
    test_flights = list(set(range(len(fps))) - set(train_flights))
    train_flights.sort()
    test_flights.sort()

    fps_train, fps_test = SplitStrList(fps, test_flights)
    fts_train, fts_test = SplitStrList(fts, test_flights)
    wcs_train, wcs_test = SplitStrList(wcs, test_flights)

    print('test flights:\n{}'.format(test_flights))
    df_trainfiles = pd.DataFrame(
        data={'flight plans': fps_train, 'flight tracks': fts_train, 'weather cubes': wcs_train})
    df_testfiles = pd.DataFrame(data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
    df_trainfiles.to_csv('train_flight_samples.txt')
    df_testfiles.to_csv('test_flight_samples.txt')
    '''

    # Uncomment block if validated & split files already exist
    df_trainfiles = pd.read_csv('train_flight_samples.txt')
    print('Loading Train Files')
    fps_train, fts_train, wcs_train, train_flights = [],[],[],[]
    fps_train, fts_train = df_trainfiles['flight plans'].values, df_trainfiles['flight tracks'].values
    wcs_train, train_flights = df_trainfiles['weather cubes'].values, list(range(len(fps_train)))
    #TODO: eliminate train_flights    
    '''

    train_dataset = CustomDataset(root_dir, fps_train, fts_train, wcs_train, ToTensor(), device='cpu')
    train_dl = DataLoader(train_dataset, collate_fn=pad_batch, batch_size=bs, num_workers=0, pin_memory=True,
                          shuffle=False, drop_last=True)

    # train_model
    '''
    model_lstm = CONV_LSTM(paradigm=paradigms[1], device=dev)
    model_gru = CONV_GRU(paradigm=paradigms[1], device=dev)
    model_indrnn = CONV_INDRNN(paradigm=paradigms[1], device=dev)
    '''
    model_lstm = CONV_RECURRENT(paradigm = paradigms[1], device=dev, rnn=torch.nn.LSTM)
    model_gru = CONV_RECURRENT(paradigm=paradigms[1], device=dev, rnn=torch.nn.GRU)
    model_indrnn = CONV_RECURRENT(paradigm=paradigms[1], device=dev, rnn=IndRNN_onlyrecurrent, rnn_layers=2)
    model_lstm_sa = CONV_RECURRENT(paradigm=paradigms[1], device=dev, rnn=torch.nn.LSTM, attn='after')
    model_gru_sa = CONV_RECURRENT(paradigm=paradigms[1], device=dev, rnn=torch.nn.GRU, attn='after')
    model_indrnn_sa = CONV_RECURRENT(paradigm=paradigms[1], device=dev, rnn=IndRNN_onlyrecurrent, rnn_layers=2, attn='after')


    mdls = [model_lstm, model_gru, model_indrnn, model_lstm_sa, model_gru_sa, model_indrnn_sa]

    for i in range(len(mdls)):
        print(mdls[i])
        sttime = datetime.now()
        print('START FIT: {}'.format(sttime))
        fit(mdls[i], train_dl, epochs, train_flights)
        mdls[i].epochs_trained = epochs
        mdls[i].save_model(batch_size=bs)
        edtime = datetime.now()
        print('DONE: {}'.format(edtime - sttime))

        if not os.path.isdir('Initialized Plots/{}'.format(mdls[i].model_name(epochs))):
            os.mkdir('Initialized Plots/{}'.format(mdls[i].model_name(epochs)))
        plots_to_move = [x for x in os.listdir('Initialized Plots') if x.__contains__('.png')]
        for plot in plots_to_move:
            shutil.move('Initialized Plots/{}'.format(plot), 'Initialized Plots/{}/{}'.format(mdls[i].model_name(epochs), plot))

        shutil.copy('test_flight_samples.txt', 'Models/{}/test_flight_samples.txt'.format(mdls[i].model_name(bs)))
        shutil.copy('train_flight_samples.txt', 'Models/{}/train_flight_samples.txt'.format(mdls[i].model_name(bs)))
        shutil.move('model_epoch_losses.txt', 'Models/{}/model_epoch_losses.txt'.format(mdls[i].model_name(bs)))


def fit(mdl: torch.nn.Module, flight_data: torch.utils.data.DataLoader, epochs: int, model_name: str = 'Default',):
    epoch_losses = torch.zeros(epochs, device=mdl.device)
    for ep in tqdm.trange(epochs, desc='epoch', position=0, leave=False):
        losses = torch.zeros(len(flight_data), device=mdl.device)

        for batch_idx, (fp, ft, wc) in enumerate(
                tqdm.tqdm(flight_data, desc='flight', position=1, leave=False)):  # was len(flight_data)
            # Extract flight plan, flight track, and weather cubes
            if mdl.device.__contains__('cuda'):
                fp = fp[:, :, 1:].cuda(device=mdl.device, non_blocking=True)
                ft = ft[:, :, 1:].cuda(device=mdl.device, non_blocking=True)
                wc = wc.cuda(device=mdl.device, non_blocking=True)
            else:
                fp, ft = fp[:, :, 1:], ft[:, :, 1:]
            if mdl.paradigm == 'Regression':
                print("\nFlight {}/{}: ".format(batch_idx + 1, len(flight_data)) + str(len(fp)) + " points")
                for pt in tqdm.trange(len(wc)):
                    mdl.optimizer.zero_grad()
                    lat, lon = fp[0][0].clone().detach(), fp[0][1].clone().detach()
                    if mdl.rnn_type == torch.nn.LSTM:
                        mdl.hidden_cell = (
                            lat.repeat(1, 1, mdl.lstm_hidden),
                            lon.repeat(1, 1, mdl.lstm_hidden))
                    elif mdl.rnn_type == torch.nn.GRU:
                        mdl.hidden_cell = torch.cat(lat.repeat(1, 1, mdl.gru_hidden / 2),
                                                    lon.repeat(1, 1, mdl.gru_hidden / 2))
                    y_pred = mdl(wc[:pt + 1], fp[:pt + 1])
                    # print(y_pred)
                    single_loss = mdl.loss_function(y_pred, ft[:pt + 1].view(-1, 2))
                    if batch_idx < len(flight_data) - 1 and pt % 50 == 0:
                        single_loss.backward()
                        mdl.optimizer.step()
                    if batch_idx == len(flight_data) - 1:
                        losses = torch.cat((losses, single_loss.view(-1)))
            elif mdl.paradigm == 'Seq2Seq':
                mdl.optimizer.zero_grad()

                # TODO: Abstraction (Unify recurrences within one class)
                # hidden cell (h_, c_) sizes: [num_layers*num_directions, batch_size, hidden_size]
                if mdl.rnn_type == torch.nn.LSTM:
                    mdl.hidden_cell = (
                        torch.repeat_interleave(fp[0, :, 0], mdl.rnn_hidden).view(-1, 1, mdl.rnn_hidden),
                        torch.repeat_interleave(fp[0, :, 1], mdl.rnn_hidden).view(-1, 1, mdl.rnn_hidden))
                elif mdl.rnn_type == torch.nn.GRU:
                    mdl.hidden_cell = torch.cat((
                        torch.repeat_interleave(fp[0, :, 0], int(mdl.rnn_hidden / 2)),
                        torch.repeat_interleave(fp[0, :, 1], int(mdl.rnn_hidden / 2))
                    )).view(1, -1, int(mdl.rnn_hidden))
                elif mdl.rnn_type == IndRNN_onlyrecurrent:
                    '''
                    #TODO: is initialization necessary for indrnn?
                    for i in range(len(mdl.rnns)):
                        mdl.rnns[i].indrnn_cell.weight_hh = torch.nn.Parameter(torch.cat((
                            torch.repeat_interleave(fp[0, :, 0], int(mdl.rnn_input / 2)),
                            torch.repeat_interleave(fp[0, :, 1], int(mdl.rnn_input / 2))
                        )))
                    '''
                y_pred = mdl(wc, fp[:])
                single_loss = mdl.loss_function(y_pred, ft)
                losses[batch_idx] = single_loss.view(-1)
                single_loss.backward()
                mdl.optimizer.step()
            if batch_idx == len(flight_data) - 1:
                epoch_losses[ep] = torch.mean(losses).view(-1)

        if ep % 10 == 0:
            if mdl.device.__contains__('cuda'):
                losses = losses.cpu()
            plt.plot(losses.detach().numpy())
            plt.title('Training (Epoch {})'.format(ep + 1))
            plt.xlabel('Flight')
            plt.ylabel('Loss (MSE)')
            # plt.savefig('Eval Epoch{}.png'.format(ep+1), dpi=400)
            plt.savefig('Initialized Plots/Eval Epoch{}.png'.format(ep + 1), dpi=400)
            plt.close()

    if mdl.device.__contains__('cuda'):
        epoch_losses = epoch_losses.cpu()
    e_losses = epoch_losses.detach().numpy()
    plt.plot(e_losses)
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.savefig('Initialized Plots/Model Eval.png', dpi=300)
    plt.close()

    plt.plot(e_losses[:])
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.ylim([0, .1])
    plt.yticks(np.linspace(0, 1, 11))
    plt.savefig('Initialized Plots/Model Eval RangeLimit.png', dpi=300)
    plt.close()

    df_eloss = pd.DataFrame({'loss': e_losses})
    df_eloss.to_csv('model_epoch_losses.txt')


if __name__ == '__main__':
    main()
