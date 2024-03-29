import torch
import os, shutil, gc
import numpy as np
import pandas as pd
import warnings
from custom_dataset import CustomDataset, ValidFiles, SplitStrList
from custom_dataset import ToTensor
from attn_dataset import ATTNDataset
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from fit import fit
from fit_attn import fit as fit_attn
from functools import partial
import ray
from ray import tune
import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch
import ray.tune.schedulers as sch
from ray.tune.suggest.bohb import TuneBOHB

def main():
    warnings.filterwarnings('ignore')
    if os.name == 'nt':
        torch.multiprocessing.set_start_method('spawn')

    # training params
    paradigms = {0: 'Regression', 1: 'Seq2Seq'}
    attns = {0: 'None', 1: 'after', 2: 'replace'}
    max_epochs = 300

    dev = 'cuda:0'
    # dev = 'cpu'
    # root_dir = '/media/lab/Local Libraries/TorchDir'
    root_dir = '/home/dualboot/Data/Train-2weeks'  # TEST DATA
    # root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'
    # Uncomment block if generating valid file & split files
    total_products = ['ECHO_TOP', 'VIL', 'uwind', 'vwind', 'tmp']
    list_products = ['ECHO_TOP']
    cube_height = 1

    flight_mins = {'KJFK_KLAX': 5 * 60, 'KIAH_KBOS': 3.5 * 60, 'KATL_KORD': 1.5 * 60,
                   'KATL_KMCO': 1.25 * 60, 'KSEA_KDEN': 2.25 * 60, 'KORD_KLGA': 1.5}
    fps, fts, wcs, dates, _ = ValidFiles(root_dir, total_products, under_min=flight_mins,
                                         fp_subdir='/Flight Plans/Sorted-interp',
                                         ft_subdir='/Flight Tracks/Interpolated')
    total_flights = len(fps)

    # Random split
    train_flights = np.random.choice(total_flights, int(total_flights * .75), replace=False)
    test_flights = list(set(range(len(fps))) - set(train_flights))
    print('Test Flights: {}'.format(test_flights))

    # cross-validation split
    '''foldstr = 'fold{}-{}'.format(fold + 1, folds)
    test_flights = list(range(int(fold * total_flights / folds), int(((fold + 1) * total_flights) / folds)))
    train_flights = list(set(range(total_flights)) - set(test_flights))
    print('fold {}/{}\t{}-{} test flights'.format(fold + 1, folds, min(test_flights), max(test_flights)))'''

    train_flights.sort()
    test_flights.sort()

    fps_train, fps_test = SplitStrList(fps, test_flights)
    fts_train, fts_test = SplitStrList(fts, test_flights)
    wcs_train, wcs_test = SplitStrList(wcs, test_flights)

    df_trainfiles = pd.DataFrame(
        data={'flight plans': fps_train, 'flight tracks': fts_train, 'weather cubes': wcs_train})
    df_testfiles = pd.DataFrame(
        data={'flight plans': fps_test, 'flight tracks': fts_test, 'weather cubes': wcs_test})
    #df_trainfiles.to_csv('tune_train_flight_samples.txt')
    #df_testfiles.to_csv('tune_test_flight_samples.txt')


    # Uncomment block if validated & split files already exist
    # df_trainfiles = pd.read_csv('/home/dualboot/Desktop/Flight-Track-Prediction/Models/ECHO_TOP/CONV1.0.05-LSTM1lay-OPTAdam-LOSSMSELoss()-EPOCHS500-BATCH1-RNN6_100_3/fold1-4/train_flight_samples.txt')
    # print('Loading Train Files')
    # fps_train, fts_train, wcs_train, train_flights = [],[],[],[]
    # fps_train, fts_train = df_trainfiles['flight plans'].values, df_trainfiles['flight tracks'].values
    # wcs_train, train_flights = df_trainfiles['weather cubes'].values, list(range(len(fps_train)))
    # wcs_train = [[x.split('\'')[1]] for x in wcs_train]
    #
    # df_testfiles = pd.read_csv('/home/dualboot/Desktop/Flight-Track-Prediction/Models/ECHO_TOP/CONV1.0.05-LSTM1lay-OPTAdam-LOSSMSELoss()-EPOCHS500-BATCH1-RNN6_100_3/fold1-4/test_flight_samples.txt')
    # print('Loading Test Files')
    # fps_test, fts_test, wcs_test, test_flights = [],[],[],[]
    # fps_test, fts_test = df_testfiles['flight plans'].values, df_testfiles['flight tracks'].values
    # wcs_test, test_flights = df_testfiles['weather cubes'].values, list(range(len(fps_test)))
    # wcs_test = [[x.split('\'')[1]] for x in wcs_test]


    train_dataset = ATTNDataset(root_dir, fps_train, fts_train, wcs_train, list_products, ToTensor(), device='cpu')
    test_dataset = ATTNDataset(root_dir, fps_test, fts_test, wcs_test, list_products, ToTensor(), device='cpu')

    ## MODEL PARAMETERS
    # Ray search space
    ray.init(_temp_dir='/media/dualboot/New Volume/NathanSchimpf/tmp')
    global_optim = torch.optim.RMSprop
    config_cnnlstm_opt = {
        # Pre-defined net params
        'name': 'CNN_LSTM-OPTFULL-RAND',
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[0], 'batch_size': 1,
        #'optim': tune.grid_search(['sgd','sgd+momentum','sgd+nesterov','adam','rmsprop','adadelta','adagrad']),
        'optim': tune.grid_search(['adam']),
        # Params to tune
        'ConvCh': [1, 28, 22], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 1000,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-8, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }
    config_cnnlstm_ovf = {
        # Pre-defined net params
        'name': 'CNN_LSTM-OVF',
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[0], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, 2, 4], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 100,
        'droprate': tune.grid_search([0.,1e-4,1e-3,1e-2,5e-2,10e-2,20e-2]), 'lr': 2e-4, 'epochs': 200 + 1,
        'weight_reg': tune.grid_search([0.,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]), 'batchnorm': tune.grid_search(['None','simple','learn'])
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_saalstm = {
        'name': 'CNN+SA_LSTM-CHDepths-CONST',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[1], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, 15, 15], 'HLs': [16],
        'RNNIn': 10, 'RNNDepth': 2, 'RNNHidden': 300,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs+1,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_sarlstm_opt = {
        'name': 'SA_LSTM-OPT',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[2], 'batch_size': 1, 'optim': tune.grid_search([torch.optim.Adam,torch.optim.RMSprop]),
        # Params to tune
        'ConvCh': [1, 31, 8], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 2, 'RNNHidden': 600,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-8, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_cnngru_optsched = {
        'name': 'CNN_GRU-OPTFULL-SCHED',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products, 'attn': attns[0], 'batch_size': 1, #'optim': tune.grid_search([torch.optim.Adam,torch.optim.RMSprop]),
        # Params to tune
        'optim': tune.grid_search(['sgd+nesterov','adam', 'rmsprop']),
        'forcelr': 0.01, 'forcemom':0.5, 'forcenest': True,
        'forcegamma': tune.grid_search([0.1,0.5,0.9]), 'forcestep':tune.grid_search([10,30,50]),
        'ConvCh': [1, 28, 22], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 650,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-8, 'batchnorm': 'None'
    }
    config_cnngru_SGD = {
        'name': 'CNN_GRU-SGD',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products, 'attn': attns[0], 'batch_size': 1, #'optim': tune.grid_search([torch.optim.Adam,torch.optim.RMSprop]),
        # Params to tune
        'optim': tune.grid_search(['sgd']),
        'forcelr': tune.grid_search([10**x for x in range(-6,0)]), 'forcemom': tune.grid_search(np.linspace(0,1,11).tolist()),
        'forcenest': tune.grid_search([False, True]),
        'ConvCh': [1, 28, 22], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 1, 'RNNHidden': 650,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-8, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }
    config_sargru_SGD = {
        'name': 'SAR_GRU-SGD',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products, 'attn': attns[2], 'batch_size': 1,
        # Params to tune
        'optim': tune.grid_search(['sgd']),
        'forcelr': tune.grid_search([10 ** x for x in range(-6, 0)]),
        'forcemom': tune.grid_search(np.linspace(0, 1, 11).tolist()),
        'forcenest': tune.grid_search([False, True]),
        'ConvCh': [1, 31, 8], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 2, 'RNNHidden': 600,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }
    config_sargru_opt = {
        'name': 'SAR_GRU-OPTFULL-RAND',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products, 'attn': attns[2], 'batch_size': 1, #'optim': tune.grid_search([torch.optim.Adam,torch.optim.RMSprop]),
        # Params to tune
        'optim': tune.grid_search(['sgd', 'adam', 'rmsprop', 'adadelta', 'adagrad']),
        #'forcelr': 0.01, 'forcemom': 0.7, 'forcenest': True,
        'ConvCh': [1, 31, 8], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 2, 'RNNHidden': 600,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }
    config_sargru_optsched = {
        'name': 'SAR_GRU-OPTFULL-SCHED',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products, 'attn': attns[2], 'batch_size': 1, #'optim': tune.grid_search([torch.optim.Adam,torch.optim.RMSprop]),
        # Params to tune
        'optim': tune.grid_search(['sgd', 'adam', 'rmsprop', 'adadelta']),
        'forcegamma': tune.grid_search([0.1, 0.5, 0.9]), 'forcestep': tune.grid_search([10, 30, 50]),
        #'forcelr': 0.01, 'forcemom': 0.7, 'forcenest': True,
        'ConvCh': [1, 31, 8], 'HLs': [16],
        'RNNIn': 6, 'RNNDepth': 2, 'RNNHidden': 600,
        'droprate': 1e-3, 'lr': 2e-4, 'epochs': max_epochs + 1,
        'weight_reg': 1e-6, 'batchnorm': 'None'
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }
    config_attn_optsched = {'name': 'ATTN-CONSTINIT', 'device': 'cuda:0', 'cube_height': 1, 'features': list_products[0],
                        'paradigm': paradigms[1], 'loss_function': torch.nn.MSELoss,
                        # adjustable hyperparams
                        'optim':tune.grid_search(['rmsprop','adam']), 'batch_size': 1, 'epochs': 300, 'weight_reg': 0.,
                        'forcegamma': .5, 'forcestep': 30, 'forcelr': .01,
                        # 'latlon_wb': {'w_q': 1.,'w_k': 1., 'w_v': 1., 'b_q': 0., 'b_k': -1., 'b_v': 0.},
                        # 'alt_wb': {'w_q': 1.,'w_k': 1., 'w_v': 1., 'b_q': 0., 'b_k': 0., 'b_v': 0.}
                            }
    for config in [config_attn_optsched]:
        #if global_optim == torch.optim.RMSprop:
        #    config['name'] = 'RMSProp-{}'.format(config['name'])
        chkdir = 'Models/Tuning/{}'.format(config['name'])
        if config == config_cnnlstm_ovf:
            smpl = 1
            gputil = 0.5
        else:
            smpl = 3
            gputil = 0.2
        if not os.path.isdir(chkdir):
            os.makedirs(chkdir)
        # train_model
        # bohb_hyperband = sch.HyperBandForBOHB(time_attr="training_iteration", max_t=100, reduction_factor=4,
        #                                       stop_last_trials=False)
        # bohb_opt = TuneBOHB(max_concurrent=4)
        # sched = sch.PopulationBasedTraining(time_attr='training_iteration',perturbation_interval=4.,
        #                                       hyperparam_mutations=config)
        # scheduler = sch.ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)
        fifo_sched = sch.FIFOScheduler()
        result = tune.run(
            tune.with_parameters(fit_attn, train_dataset=train_dataset, test_dataset=test_dataset, raytune=True,
                                 determinist=False, const=False, scale=True),
            resources_per_trial={"cpu": 4, "gpu": gputil},
            config=config,
            metric="valloss",
            name=config['name'],
            mode="min",
            #local_dir='/media/dualboot/New Volume/NathanSchimpf/Tuning',
            num_samples=smpl,
            scheduler=fifo_sched
        )
        df = result.results_df
        print(df)
        df_trainfiles.to_csv('Models/Tuning/{}_TRAINSPLIT.csv'.format(config['name']))
        df_testfiles.to_csv('Models/Tuning/{}_TESTSPLIT.csv'.format(config['name']))
        df.to_csv('Models/Tuning/{}.csv'.format(config['name']))


if __name__ == '__main__':
    main()
