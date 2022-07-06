import torch, json, os
from libmodels.CONV_RECURRENT import  CONV_RECURRENT
from libmodels.IndRNN_pytorch.IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as indrnn
from libmodels.multiheaded_attention import MultiHeadedAttention as MHA
from Utils.Misc import parseConfig

def load_model(model_path: str):
    mdlname = model_path.split(os.sep)[-1]

    # Load Model
    dicts = torch.load(model_path)
    struct = dicts['struct_dict']
    state_dict = dicts['state_dict']
    opt_dict = dicts['opt_dict']



    #TODO: NOT THIS
    # identify model struct
    mdlkey = ''
    if 'SAR' in mdlname:
        if 'LSTM' in mdlname:
            if 'RNN6_100_3' in mdlname: mdlkey = 'config_dflt_sarlstm'
            else: mdlkey = 'config_sarlstm'
        elif 'GRU' in mdlname:
            if 'RNN6_100_3' in mdlname: mdlkey = 'config_dflt_sargru'
            else: mdlkey = 'config_sargru'
    elif 'CONV' in mdlname:
        if 'LSTM' in mdlname:
            if 'RNN6_100_3' in mdlname: mdlkey = 'config_dflt_cnnlstm'
            else: mdlkey = 'config_cnnlstm'
        elif 'GRU' in mdlname:
            if 'RNN6_100_3' in mdlname: mdlkey = 'config_dflt_cnngru'
            else: mdlkey = 'config_cnngru'

    # load json config
    f = open('Utils/Configs.json')
    config = json.load(f)
    cfg = config['models'][mdlkey]
    glb_config = {key: config[key] for key in list(set(config.keys()) - set(['models']))}
    for key in glb_config:
        if not key in list(cfg.keys()): cfg[key] = config[key]
    if not 'weight_reg' in cfg.keys(): cfg['weight_reg'] = 0.
    cfg = parseConfig(cfg)
    mdl = CONV_RECURRENT(config=cfg)


    # hght = 3 if 'uwind' in struct['features'] or 'vwind' in struct['features'] or 'tmp' in struct['features'] else 1
    # struct['cube_height'] = hght
    # mdl = CONV_RECURRENT(config=struct)

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
