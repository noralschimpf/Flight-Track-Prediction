import json, torch
from ray import tune

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
     return newcfg


