import torch
from libmodels.CONV_LSTM import CONV_LSTM
from libmodels.CONV_GRU import CONV_GRU
from libmodels.CONV_INDRNN import CONV_INDRNN

def load_model(model_path: str):
    dicts = torch.load(model_path)
    struct = dicts['struct_dict']
    state_dict = dicts['state_dict']
    opt_dict = dicts['opt_dict']
    if 'CONV_LSTM' in struct['class']:
        mdl = CONV_LSTM(paradigm=struct['paradigm'],
                        conv_input=struct['conv_input'], conv_hidden=struct['conv_hidden'],
                        conv_output=struct['conv_output'],
                        dense_hidden=struct['dense_hidden'], dense_output=struct['dense_output'],
                        lstm_input=struct['lstm_input'], lstm_hidden=struct['lstm_hidden'],
                        lstm_output=struct['lstm_output'],
                        device=struct['device'], optim=struct['optim'], loss=struct['loss_fn'],
                        eptrained=dicts['epochs_trained'])
    elif 'CONV_GRU' in struct['class']:
        mdl = CONV_GRU(paradigm=struct['paradigm'], device=struct['device'],
                   conv_input=struct['conv_input'], conv_hidden=struct['conv_hidden'], conv_output=struct['conv_output'],
                   dense_hidden=struct['dense_hidden'], dense_output=struct['dense_output'],
                   gru_input=struct['gru_input'], gru_hidden=struct['gru_hidden'], gru_output=struct['gru_output'],
                   optim=struct['optim'], loss=struct['loss_fn'], eptrained=dicts['epochs_trained'])
    elif 'CONV_INDRNN' in struct['class']:
        mdl = CONV_INDRNN(paradigm=struct['paradigm'], device=struct['device'],
                  conv_input=struct['conv_input'], conv_hidden=struct['conv_hidden'], conv_output=struct['conv_output'],
                  dense_hidden=struct['dense_hidden'], dense_output=struct['dense_output'],
                  rnn_input=struct['rnn_input'], rnn_hidden=struct['rnn_hidden'], rnn_output=struct['rnn_output'],
                  optim=struct['optim'], loss=struct['loss_fn'], eptrained=dicts['epochs_trained'])
    mdl.load_state_dict(state_dict)
    mdl.optimizer.load_state_dict(opt_dict)
    mdl.epochs_trained = dicts['epochs_trained']
    return (mdl)