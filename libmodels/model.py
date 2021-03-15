import torch
import CONV_LSTM, CONV_GRU

def load_model(model_path: str):
    dicts = torch.load(model_path)
    struct = dicts['struct_dict']
    state_dict = dicts['state_dict']
    if struct['class'] == 'model.CONV_LSTM':
        mdl = CONV_LSTM(paradigm=struct['paradigm'],
                        conv_input=struct['conv_input'], conv_hidden=struct['conv_hidden'],
                        conv_output=struct['conv_output'],
                        dense_hidden=struct['dense_hidden'], dense_output=struct['dense_output'],
                        lstm_input=struct['lstm_input'], lstm_hidden=struct['lstm_hidden'],
                        lstm_output=struct['lstm_output'], device=struct['device'])
    elif struct['class'] == 'model.CONV_GRU':
        mdl = CONV_GRU(paradigm=struct['paradigm'], device=struct['device'],
                       conv_input=struct['conv_input'], conv_hidden=struct['conv_hidden'], conv_output=struct['conv_output'],
                       dense_hidden=struct['dense_hidden'], dense_output=struct['dense_output'],
                       gru_input=struct['gru_input'], gru_hidden=struct['gru_hidden'], gru_output=struct['gru_output'])
    mdl.load_state_dict(state_dict)
    return (mdl)