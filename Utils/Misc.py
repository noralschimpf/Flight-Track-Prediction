import json
from ray import tune
from torch import nn, optim

def str_to_list(strlist: str, type):
    tmp = strlist[1:-1].replace(',',' ')
    tmplist = tmp.split(' ')
    if strlist[0] == '[':
        retlist = [type(x) for x in tmplist if x]
        return retlist
    elif strlist[0] == '(':
        retlist = (type(x) for x in tmplist if x)
        return retlist

def parseargs(args: str):
    parsed = ()
    for arg in args.split(','):
        if '[' or '(' in arg: arg = str_to_list(arg)
        elif '{' in arg: arg = json.loads(arg)
        parsed = parsed + (arg,)
    return parsed


def parseConfig(cfg: dict):
    newcfg = {}
    for key in cfg:
        if isinstance(cfg[key], str):
            if 'tune.' in cfg[key]:
                call = cfg[key].split('.')
                func, args = call.split('(')[0], call.split('(')[1][:-1]
                parseargs(args)
                if func == 'uniform': newcfg[key] = tune.uniform(args)
                elif func == 'quniform': newcfg[key] = tune.quniform(args)
                elif func == 'loguniform': newcfg[key] = tune.loguniform(args)
                elif func == 'qloguniform': newcfg[key] = tune.qloguniform(args)
                elif func == 'randn': newcfg[key] = tune.randn(args)
                elif func == 'qrandn': newcfg[key] = tune.qrandn(args)
                elif func == 'randint': newcfg[key] = tune.randint(args)
                elif func == 'qrandint': newcfg[key] = tune.qrandint(args)
                elif func == 'lograndint': newcfg[key] = tune.lograndint(args)
                elif func == 'logqrandint': newcfg[key] = tune.logqrandint(args)
                elif func == 'choice': newcfg[key] = tune.choice(args)
                elif func == 'grid_search': newcfg[key] = tune.grid_search([args])
                elif func == 'sample_from':
                    raise Exception('custom distributions are not currently supported')
            elif 'torch.' in cfg[key]:
                split = cfg[key].split('.')
                if split[1] == 'nn':
                    if split[2] == 'LSTM': newcfg[key] = nn.LSTM
                    elif split[2] == 'GRU': newcfg[key] = nn.GRU
                    elif split[2] == 'RNN': newcfg[key] = nn.RNN

                    elif split[2] == 'L1Loss': newcfg[key] = nn.L1Loss
                    elif split[2] == 'MSELoss': newcfg[key] = nn.MSELoss
                    elif split[2] == 'SmoothL1Loss': newcfg[key] = nn.SmoothL1Loss
                    elif split[2] == 'MSELoss': newcfg[key] = nn.MSELoss


                elif split[1] == 'optim':
                    if split[2] == 'Adadelta': newcfg[key] = optim.Adadelta
                    elif split[2] == 'Adam': newcfg[key] = optim.Adam
                    elif split[2] == 'AdamW': newcfg[key] = optim.AdamW
                    elif split[2] == 'SparseAdam': newcfg[key] = optim.SparseAdam
                    elif split[2] == 'Adamax': newcfg[key] = optim.Adamax
                    elif split[2] == 'ASGD': newcfg[key] = optim.ASGD
                    elif split[2] == 'LBFGS': newcfg[key] = optim.LBFGS
                    elif split[2] == 'NAdam': newcfg[key] = optim.NAdam
                    elif split[2] == 'RAdam': newcfg[key] = optim.RAdam
                    elif split[2] == 'RMSProp': newcfg[key] = optim.RMSProp
                    elif split[2] == 'RProp': newcfg[key] = optim.RProp
                    elif split[2] == 'SGD': newcfg[key] = optim.SGD
                    else: raise Exception('No optimizer specified')
            else: newcfg[key] = cfg[key]
        else: newcfg[key] = cfg[key]

    return newcfg


