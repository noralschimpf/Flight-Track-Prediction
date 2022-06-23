import torch
import os, shutil
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from custom_dataset import CustomDataset, ValidFiles, SplitStrList, pad_batch
from attn_dataset import ATTNDataset, pad_batch as attn_pad
from custom_dataset import ToTensor
from fit import fit
from fit_attn import fit as ATTNfit
from Utils.Misc import str_to_list, parseConfig
from global_vars import flight_mins, flight_min_tol
import json


'''TRAINING CONFIGS
3D EXPECTED: 1000 Epochs, Batch Size 64 -> MSE .001 Train/Test 90/10
3D EXPECTED: 500 Epochs -> MSE .001 (no batching)   Train/Test 75/25
4D EXPECTED: 500 Epochs -> MSE .03                  Train/Test 75/25

'''


def main():
    warnings.filterwarnings('ignore')
    if os.name == 'nt':
        torch.multiprocessing.set_start_method('spawn')

    # Open Training Config
    f = open('Utils/Configs.json')
    config = json.load(f)




    # Uncomment block if generating valid file & split files
    total_flights, fps, fts, wcs, dates = -1, [], [], [], []
    if config['data'] == 'generate':
        total_products=['ECHO_TOP','VIL','uwind','vwind','tmp']
        list_products = [['ECHO_TOP']]; cube_height = 1
        fps, fts, wcs, dates, _ = ValidFiles(config['root_dir'], total_products, under_min=flight_min_tol,
                         fp_subdir='/Flight Plans/Sorted-interp', ft_subdir='/Flight Tracks/Interpolated')
        total_flights = len(fps)

    #cnnlstm = tune.Analysis('~/ray_results/RMSProp-CNN_LSTM-CHDepths')
    #cnngru = tune.Analysis('~/ray_results/CNN_GRU')
    #cnnindrnn = tune.Analysis('~/ray_results/CNN_IndRNN')
    #saalstm = tune.Analysis('~/ray_results/CNN+SA_LSTM')
    #cfg_lstm = sarlstm.get_best_config(metric='valloss', mode='min')

    cfgs = [config['models']['config_dflt_cnnlstm'],
            config['models']['config_cnnlstm']]

    # Correct Models
    for cfg in cfgs:
        cfg['device'] = config['dev']
        cfg['epochs'] = config['epochs']
        cfg['batch_size'] = config['bs']
        if not 'weight_reg' in cfg.keys():
            cfg['weight_reg'] = 0.
        '''if isinstance(config['optim'], str):
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
        if isinstance(config['ConvCh'], str): config['ConvCh'] = str_to_list(config['ConvCh'], int)'''


    for products in config['list_products']:
        cube_height = 3 if 'uwind' in products or 'vwind' in products or 'tmp' in products else 1
        prdstr = '&'.join(products)
        for fold in range(config['folds']):
            train_flights, test_flights = [], []
            trainname, testname = 'train_flight_samples.txt', 'test_flight_samples.txt'
            # Random split
            if config['split'] == 'random':
                train_flights = np.random.choice(total_flights, int(total_flights * .75), replace=False)
                test_flights = list(set(range(len(fps))) - set(train_flights))

            # cross-validation split
            elif config['split'] == 'xval':
                foldstr = 'fold{}-{}'.format(fold+1,config['folds'])
                trainname, testname = f'{foldstr} {trainname}', f'{foldstr} {testname}'
                test_flights = list(range( int(fold*total_flights/config['folds']), int(((fold+1)*total_flights)/config['folds']) ))
                train_flights = list(set(range(total_flights)) - set(test_flights))
                print('fold {}/{}\t{}-{} test flights'.format(fold+1,config['folds'],min(test_flights),max(test_flights)))
                train_flights.sort()
                test_flights.sort()

            if config['folds'] == 1:
                train_flights = list(range(total_flights))
                test_flights = []

            fps_train, fps_test = SplitStrList(fps, test_flights)
            fts_train, fts_test = SplitStrList(fts, test_flights)
            wcs_train, wcs_test = SplitStrList(wcs, test_flights)

            df_trainfiles = pd.DataFrame(
                data={'flight plans': fps_train, 'flight tracks': fts_train, 'weather cubes': wcs_train})
            df_testfiles = pd.DataFrame(data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
            df_trainfiles.to_csv(trainname)
            df_testfiles.to_csv(testname)



            # # Uncomment block if validated & split files already exist
            if config['data'] == 'read':
                df_trainfiles = pd.read_csv(config['train_splitfile'])
                print('Loading Train Files')
                fps_train, fts_train, wcs_train, train_flights = [],[],[],[]
                fps_train, fts_train = df_trainfiles['flight plans'].values, df_trainfiles['flight tracks'].values
                wcs_train, train_flights = df_trainfiles['weather cubes'].values, list(range(len(fps_train)))

                df_testfiles = pd.read_csv(config['test_splitfile'])
                print('Loading Test Files')
                fps_test, fts_test, wcs_test, test_flights = [], [], [], []
                fps_test, fts_test = df_testfiles['flight plans'].values, df_testfiles['flight tracks'].values
                wcs_test, test_flights = df_testfiles['weather cubes'].values, list(range(len(fps_test)))
                foldstr = 'fold1-1'

            train_dataset = CustomDataset(config['root_dir'], fps_train, fts_train, wcs_train, products, ToTensor(), device='cpu')
            train_dataset = CustomDataset(config['root_dir'], fps_train, fts_train, wcs_train, products, ToTensor(), device='cpu')
            test_dataset = CustomDataset(config['root_dir'], fps_test, fts_test, wcs_test, products, ToTensor(), device='cpu')

            # train_model
            for config in cfgs:
                mdl = fit(config, train_dataset, test_dataset, raytune=False, determinist=False, const=False, gradclip=True, scale=True)
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
                                    os.path.join('Models',prdstr, mdl.model_name(), foldstr, fname))

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


if __name__ == '__main__':
    main()
