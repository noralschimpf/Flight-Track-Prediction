import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from datetime import datetime
from custom_dataset import CustomDataset, ValidFiles, SplitStrList, pad_batch
from custom_dataset import ToTensor
from model import CONV_LSTM
from torch.utils.data import DataLoader

# TODO: BATCH FLIGHTS
'''TRAINING CONFIGS
3D EXPECTED: 1000 Epochs, Batch Size 64 -> MSE .001 Train/Test 90/10
3D EXPECTED: 500 Epochs -> MSE .001 (no batching)   Train/Test 75/25
4D EXPECTED: 500 Epochs -> MSE .03                  Train/Test 75/25

'''
def main():
    torch.multiprocessing.set_start_method('spawn')

    # training params
    epochs = 5
    bs = 1
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}

    dev = 'cuda:1'
    # dev = 'cpu'
    # root_dir = '/media/lab/Local Libraries/TorchDir'
    root_dir = 'data/' # TEST DATA
    # root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'

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
    df_testfiles = pd.DataFrame(data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
    df_testfiles.to_csv('test_flight_samples.txt')

    train_dataset = CustomDataset(root_dir, fps_train, fts_train, wcs_train, ToTensor(), device='cpu')
    train_dl = DataLoader(train_dataset, collate_fn=pad_batch, batch_size=bs, num_workers=8, pin_memory=True, shuffle=False, drop_last=True)

    # train_model
    model = CONV_LSTM(paradigm=paradigms[1], device=dev)
    sttime = datetime.now()
    print('START FIT: {}'.format(sttime))
    fit(model, train_dl, epochs, train_flights)
    model.save_model(opt='Adam', epochs=epochs, batch_size=bs)
    edtime = datetime.now()
    print('DONE: {}'.format(edtime - sttime))


def fit(mdl: torch.nn.Module, flight_data: torch.utils.data.DataLoader, epochs: int, model_name: str = 'Default'):
    epoch_losses = torch.zeros(epochs, device=mdl.device)
    for ep in tqdm.trange(epochs, desc='epoch', position=0, leave=False):
        losses = torch.zeros(len(flight_data), device=mdl.device)

        for batch_idx, (fp, ft, wc) in enumerate(tqdm.tqdm(flight_data, desc='flight', position=1, leave=False)):  # was len(flight_data)
            # Extract flight plan, flight track, and weather cubes
            if mdl.device.__contains__('cuda'):
                fp= fp[:, :, 1:].cuda(device=mdl.device, non_blocking=True)
                ft = ft[:, :, 1:].cuda(device=mdl.device, non_blocking=True)
                wc = wc.cuda(device=mdl.device, non_blocking=True)
            else:
                fp, ft = fp[:,:,1:], ft[:,:,1:]
            if mdl.paradigm == 'Regression':
                print("\nFlight {}/{}: ".format(batch_idx + 1, len(flight_data)) + str(len(fp)) + " points")
                for pt in tqdm.trange(len(wc)):
                    mdl.optimizer.zero_grad()
                    lat, lon = fp[0][0].clone().detach(), fp[0][1].clone().detach()
                    mdl.hidden_cell = (
                        lat.repeat(1, 1, mdl.lstm_hidden),
                        lon.repeat(1, 1, mdl.lstm_hidden))
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

                # hidden cell (h_, c_) sizes: [num_layers*num_directions, batch_size, hidden_size]
                mdl.hidden_cell = (
                    torch.repeat_interleave(fp[:, 0, 0], mdl.lstm_hidden).view(1, -1, mdl.lstm_hidden),
                    torch.repeat_interleave(fp[:, 0, 1], mdl.lstm_hidden).view(1, -1, mdl.lstm_hidden))

                y_pred = mdl(wc, fp[:])
                single_loss = mdl.loss_function(y_pred, ft[:])
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

    plt.plot(e_losses[3:])
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.savefig('Initialized Plots/Model Eval.png', dpi=300)

if __name__== '__main__':
    main()