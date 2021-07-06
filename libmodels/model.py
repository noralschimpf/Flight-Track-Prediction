import torch
from libmodels.CONV_RECURRENT import  CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.multiheaded_attention import MultiHeadedAttention as MHA

def load_model(model_path: str):
    dicts = torch.load(model_path)
    struct = dicts['struct_dict']
    state_dict = dicts['state_dict']
    opt_dict = dicts['opt_dict']
    #TODO: cube_height=struct['cube height']
    hght = 3 if 'uwind' in struct['features'] or 'vwind' in struct['features'] or 'tmp' in struct['features'] else 1
    mdl = CONV_RECURRENT(paradigm=struct['paradigm'], device=struct['device'], cube_height=hght,
                         conv_input=struct['conv_input'], conv_hidden=struct['conv_hidden'], conv_output=struct['conv_output'],
                         dense_hidden=struct['dense_hidden'], rnn= struct['rnn_type'], rnn_layers=struct['rnn_layers'],
                         rnn_input=struct['rnn_input'], rnn_hidden=struct['rnn_hidden'], rnn_output=struct['rnn_output'],
                         attn=struct['attntype'],droprate=struct['droprate'], features=struct['features'], batchnorm=struct['batchnorm'],
                         optim=struct['optim'], loss=struct['loss_fn'], eptrained=dicts['epochs_trained'], batch_size=dicts['batch_size']
                         )
    mdl.load_state_dict(state_dict)
    mdl.optimizer.load_state_dict(opt_dict)
    mdl.epochs_trained = dicts['epochs_trained']
    return (mdl)

@torch.no_grad()
def init_constant(net, val=0.5):
    if type(net) == torch.nn.Linear:
        net.weight.fill_(val)
        if not net.bias == None: net.bias.fill_(val)
    elif type(net) == torch.nn.Conv2d:
        net.weight.fill_(val)
        net.bias.fill_(val)
    elif type(net) == torch.nn.Conv3d:
        net.weight.fill_(val)
        net.bias.fill_(val)
    # elif type(net) == MHA:
    #     net.weight.fill_(val)
    #     net.bias.fill_(val)
    elif type(net) == torch.nn.LSTM or \
            type(net) == torch.nn.GRU or \
            type(net) == indrnn:
        for name, param in net.named_parameters():
            if 'bias' in name:
                param.fill_(val)
            elif 'weight' in name:
                param.fill_(val)
