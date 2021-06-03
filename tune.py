import torch
import os, shutil, gc
import numpy as np
import pandas as pd
import warnings
from custom_dataset import CustomDataset, ValidFiles, SplitStrList
from custom_dataset import ToTensor
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as cuda_indrnn
from fit import fit
from functools import partial
import ray
from ray import tune
import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining as PBT

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
    root_dir = 'data/'  # TEST DATA
    # root_dir = 'D:/NathanSchimpf/Aircraft-Data/TorchDir'
    # Uncomment block if generating valid file & split files
    total_products = ['ECHO_TOP', 'VIL', 'uwind', 'vwind', 'tmp']
    list_products = ['ECHO_TOP']
    cube_height = 1

    fps, fts, wcs, dates, _ = ValidFiles(root_dir, total_products, under_min=100)
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
    df_trainfiles.to_csv('tune_train_flight_samples.txt')
    df_testfiles.to_csv('tune_test_flight_samples.txt')

    '''
    # Uncomment block if validated & split files already exist
    df_trainfiles = pd.read_csv('tune_train_flight_samples.txt')
    print('Loading Train Files')
    fps_train, fts_train, wcs_train, train_flights = [],[],[],[]
    fps_train, fts_train = df_trainfiles['flight plans'].values, df_trainfiles['flight tracks'].values
    wcs_train, train_flights = df_trainfiles['weather cubes'].values, list(range(len(fps_train)))
    
    df_testfiles = pd.read_csv('tune_test_flight_samples.txt')    
    print('Loading Test Files')
    fps_test, fts_test, wcs_train, train_flights = [],[],[],[]
    fps_test, fts_test = df_testfiles['flight plans'].values, df_testfiles['flight tracks'].values
    wcs_test, test_flights = df_testfiles['weather cubes'].values, list(range(len(fps_test)))
    '''

    train_dataset = CustomDataset(root_dir, fps_train, fts_train, wcs_train, list_products, ToTensor(), device='cpu')
    test_dataset = CustomDataset(root_dir, fps_test, fts_test, wcs_test, list_products, ToTensor(), device='cpu')

    ## MODEL PARAMETERS
    # Ray search space
    ray.init(num_gpus=2)
    global_optim = torch.optim.Adam
    config_cnnlstm = {
        # Pre-defined net params
        'name': 'CNN_LSTM',
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[0], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, tune.randint(1, 25), tune.randint(1, 25)], 'HLDepth': tune.randint(1, 3),
        'HLs': tune.sample_from(lambda spec: np.random.random_integers(1, high=33, size=spec.config.HLDepth)),
        'RNNIn': tune.randint(3, 21), 'RNNDepth': tune.randint(0, 5), 'RNNHidden': tune.randint(10, 501),
        'droprate': tune.uniform(0, 0.3), 'lr': tune.loguniform(2e-4, 2e-2), 'epochs': tune.randint(1, max_epochs + 1),
        'weight_reg': tune.loguniform(2e-5,2e-1), 'batchnorm': tune.choice(['None','simple','learn'])
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_saalstm = {
        'name': 'CNN+SA_LSTM',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[1], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, tune.randint(1, 25), tune.randint(1, 25)], 'HLDepth': tune.randint(1, 3),
        'HLs': tune.sample_from(lambda spec: np.random.random_integers(1, high=33, size=spec.config.HLDepth)),
        'RNNIn': tune.randint(3, 21), 'RNNDepth': tune.randint(0, 5), 'RNNHidden': tune.randint(10, 501),
        'droprate': tune.uniform(0, 0.3), 'lr': tune.loguniform(2e-4, 2e-2), 'epochs': tune.randint(1, max_epochs + 1),
        'weight_reg': tune.loguniform(2e-5, 2e-1), 'batchnorm': tune.choice(['None', 'simple', 'learn'])
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_sarlstm = {
        'name': 'SA_LSTM',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.LSTM,
        'features': list_products, 'attn': attns[2], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, tune.randint(1, 25), tune.randint(1, 25)], 'HLDepth': tune.randint(1, 3),
        'HLs': tune.sample_from(lambda spec: np.random.random_integers(1, high=33, size=spec.config.HLDepth)),
        'RNNIn': tune.randint(3, 21), 'RNNDepth': tune.randint(0, 5), 'RNNHidden': tune.randint(10, 501),
        'droprate': tune.uniform(0, 0.3), 'lr': tune.loguniform(2e-4, 2e-2), 'epochs': tune.randint(1, max_epochs + 1),
        'weight_reg': tune.loguniform(2e-5, 2e-1), 'batchnorm': tune.choice(['None', 'simple', 'learn'])
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_cnngru = {
        'name': 'CNN_GRU',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': torch.nn.GRU,
        'features': list_products, 'attn': attns[0], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, tune.randint(1, 25), tune.randint(1, 25)], 'HLDepth': tune.randint(1, 3),
        'HLs': tune.sample_from(lambda spec: np.random.random_integers(1, high=33, size=spec.config.HLDepth)),
        'RNNIn': tune.randint(3, 21), 'RNNDepth': tune.randint(0, 5), 'RNNHidden': tune.randint(10, 501),
        'droprate': tune.uniform(0, 0.3), 'lr': tune.loguniform(2e-4, 2e-2), 'epochs': tune.randint(1, max_epochs + 1),
        'weight_reg': tune.loguniform(2e-5, 2e-1), 'batchnorm': tune.choice(['None', 'simple', 'learn'])
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    config_cnnindrnn = {
        'name': 'CNN_IndRNN',
        # Pre-defined net params
        'paradigm': paradigms[1], 'cube_height': cube_height, 'device': dev, 'rnn': indrnn,
        'features': list_products, 'attn': attns[0], 'batch_size': 1, 'optim': global_optim,
        # Params to tune
        'ConvCh': [1, tune.randint(1, 25), tune.randint(1, 25)], 'HLDepth': tune.randint(1, 3),
        'HLs': tune.sample_from(lambda spec: np.random.random_integers(1, high=33, size=spec.config.HLDepth)),
        'RNNIn': tune.randint(3, 21), 'RNNDepth': tune.randint(1, 5), 'RNNHidden': tune.randint(10, 501),
        'droprate': tune.uniform(0, 0.3), 'lr': tune.loguniform(2e-6, 2e-2), 'epochs': tune.randint(1, max_epochs + 1),
        'weight_reg': tune.loguniform(2e-5, 2e-1), 'batchnorm': tune.choice(['None', 'simple', 'learn'])
        # 'optim': tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]),
    }

    #for config in [config_cnnlstm, config_saalstm, config_sarlstm, config_cnngru, config_cnnindrnn]:
    for config in [config_cnnlstm, config_saalstm, config_sarlstm, config_cnngru]:
        if global_optim == torch.optim.RMSprop:
            config['name'] = 'RMSProp-{}'.format(config['name'])
        chkdir = 'Models/Tuning/{}'.format(config['name'])
        if not os.path.isdir(chkdir):
            os.makedirs(chkdir)
        # train_model
        scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)
        result = tune.run(
            tune.with_parameters(fit, train_dataset=train_dataset, test_dataset=test_dataset, raytune=True),
            resources_per_trial={"cpu": 6, "gpu": 1},
            config=config,
            metric="valloss",
            name=config['name'],
            mode="min",
            num_samples=150,
            scheduler=scheduler,
        )
        df = result.results_df
        print(df)
        df.to_csv('Models/Tuning/{}.csv'.format(config['name']))
        #fit(config, train_dataset, test_dataset, checkpoint_dir='Models/Tuning/CNN-LSTM1lay')




if __name__ == '__main__':
    main()