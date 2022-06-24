import torch
import os, shutil, gc, logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import tqdm
from datetime import datetime
from custom_dataset import CustomDataset, ValidFiles, SplitStrList, pad_batch
from attn_dataset import ATTNDataset, pad_batch as attn_pad
from custom_dataset import ToTensor
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
#from libmodels.IndRNN_pytorch.cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from torch.utils.data import DataLoader
from fit import fit
from fit_attn import fit as ATTNfit
from ray import tune
from Utils.Misc import str_to_list
from global_vars import flight_mins, flight_min_tol


'''TRAINING CONFIGS
3D EXPECTED: 1000 Epochs, Batch Size 64 -> MSE .001 Train/Test 90/10
3D EXPECTED: 500 Epochs -> MSE .001 (no batching)   Train/Test 75/25
4D EXPECTED: 500 Epochs -> MSE .03                  Train/Test 75/25

'''


def main():
    logging.basicConfig(filename='logs/Training.log', filemode='w', level=logging.info() )
    warnings.filterwarnings('ignore')
    if os.name == 'nt':
        torch.multiprocessing.set_start_method('spawn')

    # training params
    epochs = 500
    bs = 1
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}
    attns = {0: 'None', 1: 'after', 2: 'replace'}
    folds = 1

    dev = 'cuda:0'
    #dev = 'cpu'
    root_dir = '/home/dualboot/Data/Train'


    ## MODEL PARAMETERS
    #atts = ['None','after', 'replace']
    #atts = ['None']
    #recur_types = [torch.nn.LSTM, torch.nn.GRU, indrnn]
    #recur_types = [torch.nn.LSTM]
    #rnn_lays = [1]
    #drop = 0.0



    # Uncomment block if generating valid file & split files
    total_products=['ECHO_TOP','VIL','uwind','vwind','tmp']
    #list_products=[['ECHO_TOP'], ['VIL'],['tmp'],['vwind'],['uwind']]
    list_products = [['ECHO_TOP']]; cube_height = 1
    # flight_mins = {'KJFK_KLAX': 5*60, 'KIAH_KBOS': 3.5*60, 'KATL_KORD': 1.5*60,
    #                'KATL_KMCO': 1.25*60, 'KSEA_KDEN': 2.25*60, 'KORD_KLGA': 1.5}
    fps, fts, wcs, dates, _ = ValidFiles(root_dir, total_products, under_min=flight_min_tol,
                     fp_subdir='/Flight Plans/Sorted-interp', ft_subdir='/Flight Tracks/Interpolated')
    total_flights = len(fps)

    #cnnlstm = tune.Analysis('~/ray_results/RMSProp-CNN_LSTM-CHDepths')
    #cnngru = tune.Analysis('~/ray_results/CNN_GRU')
    #cnnindrnn = tune.Analysis('~/ray_results/CNN_IndRNN')
    #saalstm = tune.Analysis('~/ray_results/CNN+SA_LSTM')
    #cfg_lstm = sarlstm.get_best_config(metric='valloss', mode='min')

    config_cnngru_optsched = {
        'name': 'CNN_GRU-OPTFULL-SCHED',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products[0], 'attn': attns[0], 'batch_size': bs, #'optim': tune.grid_search([torch.optim.Adam,torch.optim.RMSprop]),
        # Params to tune
        'optim': tune.grid_search(['sgd+nesterov','adam', 'rmsprop']),
        'forcelr': 0.01, 'forcemom':0.5, 'forcenest': True,
        'forcegamma': 0.5, 'forcestep':10,
        'ConvCh': [1, 28, 22], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 650,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': 30,
        'weight_reg': 1e-8, 'batchnorm': 'None'
    }

    config_cnnlstm = {
        # Pre-defined net params
        'name': 'CNN_LSTM-TUNED',
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products[0], 'attn': attns[0], 'batch_size': bs,
        'optim': 'adam', 'forcegamma': .5, 'forcestep': 10, 'forcelr': .01,
        # Params to tune
        'ConvCh': [1, 28, 22], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 1000,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_sarlstm = {
        'name': 'SA_LSTM-TUNED',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products[0], 'attn': attns[2], 'batch_size': bs,
        'optim': 'adam', 'forcegamma': .5, 'forcestep': 30, 'forcelr': .01,
        # Params to tune
        'ConvCh': [1, 31, 8], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 2, 'RNNHidden': 600,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_cnngru = {
        'name': 'CNN_GRU-TUNED',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products[0], 'attn': attns[0], 'batch_size': bs,
         #'optim': torch.optim.RMSprop,
        'optim': 'rmsprop', 'forcegamma': 0.5, 'forcestep': 10, 'forcelr': 0.01,
        # Params to tune
        'ConvCh': [1, 28, 22], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 650,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_sargru = {
        'name': 'SA_GRU-TUNED',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products[0], 'attn': attns[2], 'batch_size': bs,
        #'optim': torch.optim.RMSprop,
        'optim': 'rmsprop', 'forcegamma': 0.5, 'forcestep': 30, 'forcelr': 0.01,
        # Params to tune
        'ConvCh': [1, 31, 8], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 2, 'RNNHidden': 600,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_dflt_cnnlstm = {
        # Pre-defined net params
        'name': 'CNN_LSTM-DFLT',
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products[0], 'attn': attns[0], 'batch_size': bs,
        'optim': '', #'optim': torch.optim.Adam,
        # Params to tune
        'ConvCh': [1, 2, 4], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 100,
        'droprate': 0., 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 0., 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_dflt_sarlstm = {
        'name': 'SA_LSTM-DFLT',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products[0], 'attn': attns[2], 'batch_size': bs, 'optim': torch.optim.Adam,
        # Params to tune
        'ConvCh': [1, 2, 4], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 100,
        'droprate': 0., 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 0., 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_dflt_cnngru = {
        'name': 'CNN_GRU-DFLT',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products[0], 'attn': attns[0], 'batch_size': bs, 'optim': torch.optim.Adam,
        # Params to tune
        'ConvCh': [1, 2, 4], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 100,
        'droprate': 0., 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 0., 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_dflt_sargru = {
        'name': 'SA_GRU-DFLT',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products[0], 'attn': attns[2], 'batch_size': bs, 'optim': torch.optim.Adam,
        # Params to tune
        'ConvCh': [1, 2, 4], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 100,
        'droprate': 0., 'lr': 2e-4, 'epochs': epochs,
        'weight_reg': 0., 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_attn_init = {'name': 'ATTN-CONSTINIT', 'device': 'cuda:0', 'cube_height': 1, 'features': list_products[0],
                        'paradigm': paradigms[1], 'loss_function': torch.nn.MSELoss,
                        # adjustable hyperparams
                        'optim':'rmsprop', 'batch_size': bs, 'epochs': epochs,
                        'forcegamma': .5, 'forcestep': 30, 'forcelr': .01,
                        'latlon_wb': {'w_q': 1.,'w_k': 1., 'w_v': 1., 'b_q': 0., 'b_k': -1., 'b_v': 0.},
                        'alt_wb': {'w_q': 1.,'w_k': 1., 'w_v': 1., 'b_q': 0., 'b_k': 0., 'b_v': 0.}}

    # cfgs = [config_cnngru_optsched]
    # cfgs = [config_cnngru, config_cnnlstm, config_sargru, config_sarlstm]
    #cfgs = [config_cnngru, config_cnnlstm, config_sarlstm]
    # cfgs = [config_attn_init]
    cfgs = [config_dflt_cnnlstm, config_cnnlstm]
    # Correct Models
    for config in cfgs:
        config['device'] = dev
        if not 'weight_reg' in config.keys():
            config['weight_reg'] = 0.
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


    for products in list_products:
        cube_height = 3 if 'uwind' in products or 'vwind' in products or 'tmp' in products else 1
        prdstr = '&'.join(products)
        for fold in range(folds):
            # Random split
            #train_flights = np.random.choice(total_flights, int(total_flights * .75), replace=False)
            #test_flights = list(set(range(len(fps))) - set(train_flights))

            # cross-validation split
            foldstr = 'fold{}-{}'.format(fold+1,folds)
            test_flights = list(range( int(fold*total_flights/folds), int(((fold+1)*total_flights)/folds) ))
            train_flights = list(set(range(total_flights)) - set(test_flights))
            print('fold {}/{}\t{}-{} test flights'.format(fold+1,folds,min(test_flights),max(test_flights)))
            train_flights.sort()
            test_flights.sort()

            if folds == 1:
                train_flights = list(range(total_flights))
                test_flights = []

            fps_train, fps_test = SplitStrList(fps, test_flights)
            fts_train, fts_test = SplitStrList(fts, test_flights)
            wcs_train, wcs_test = SplitStrList(wcs, test_flights)

            df_trainfiles = pd.DataFrame(
                data={'flight plans': fps_train, 'flight tracks': fts_train, 'weather cubes': wcs_train})
            df_testfiles = pd.DataFrame(data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
            df_trainfiles.to_csv('train_flight_samples.txt'.format(foldstr))
            df_testfiles.to_csv('test_flight_samples.txt'.format(foldstr))



            # # Uncomment block if validated & split files already exist
            # df_trainfiles = pd.read_csv('/home/dualboot/Desktop/Flight-Track-Prediction/Models/ECHO_TOP/CONV1.0.05-LSTM1lay-OPTAdam-LOSSMSELoss()-EPOCHS500-BATCH1-RNN6_100_3/fold1-4/train_flight_samples.txt')
            # print('Loading Train Files')
            # fps_train, fts_train, wcs_train, train_flights = [],[],[],[]
            # fps_train, fts_train = df_trainfiles['flight plans'].values, df_trainfiles['flight tracks'].values
            # wcs_train, train_flights = df_trainfiles['weather cubes'].values, list(range(len(fps_train)))
            #
            # df_testfiles = pd.read_csv(
            #     '/home/dualboot/Desktop/Flight-Track-Prediction/Models/ECHO_TOP/CONV1.0.05-LSTM1lay-OPTAdam-LOSSMSELoss()-EPOCHS500-BATCH1-RNN6_100_3/fold1-4/test_flight_samples.txt')
            # print('Loading Test Files')
            # fps_test, fts_test, wcs_test, test_flights = [], [], [], []
            # fps_test, fts_test = df_testfiles['flight plans'].values, df_testfiles['flight tracks'].values
            # wcs_test, test_flights = df_testfiles['weather cubes'].values, list(range(len(fps_test)))
            # foldstr = 'fold1-0'

            train_dataset = CustomDataset(root_dir, fps_train, fts_train, wcs_train, products, ToTensor(), device='cpu')
            train_dataset = CustomDataset(root_dir, fps_train, fts_train, wcs_train, products, ToTensor(), device='cpu')
            test_dataset = CustomDataset(root_dir, fps_test, fts_test, wcs_test, products, ToTensor(), device='cpu')

            # train_model
            for config in cfgs:
                # retry fit if diverge
                retries=3; retry_count=0
                for i in range(retries):
                    try: mdl = fit(config, train_dataset, test_dataset, raytune=False, determinist=False, const=False,
                                   gradclip=True, scale=True)
                    except ValueError as e:
                        if i < retries:
                            newlr = config['lr']/10
                            config['lr'] = newlr
                            retry_count+=1
                            logging.warning(e)
                            logging.info(f'reducing learning rate to {newlr}')
                        else:
                            logging.warning(e)
                            logging.error(f'Cannot converge {mdl.model_name()}')
                if not retry_count==retries: foldstr = 'IGNORE' + foldstr

                mdl.epochs_trained = config['epochs']
                mdl.save_model(override=True)
                shutil.rmtree('Models/{}/{}'.format(prdstr, mdl.model_name().replace(f'EPOCHS{mdl.epochs_trained}', 'EPOCHS0')))
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


if __name__ == '__main__':
    main()
