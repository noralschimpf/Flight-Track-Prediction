import torch

import numpy as np
import pandas as pd
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
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    dev = 'cuda:0'
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
    train_dl = DataLoader(train_dataset, collate_fn=pad_batch, batch_size=1, num_workers=8, pin_memory=True, shuffle=False, drop_last=True)
    # train_model
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}
    model = CONV_LSTM(paradigm=paradigms[1], device=dev)
    # set training epochs and train
    epochs = 20
    sttime = datetime.now()

    print('START FIT: {}'.format(sttime))
    model.fit(train_dl, epochs, train_flights)
    model.save_model(opt='Adam', epochs=epochs)

    edtime = datetime.now()
    print('DONE: {}'.format(edtime - sttime))
