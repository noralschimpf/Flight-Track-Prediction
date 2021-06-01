def str_to_list(strlist: str, type):
    tmp = strlist[1:-1].replace(',',' ')
    tmplist = tmp.split(' ')
    retlist = [type(x) for x in tmplist if x]
    return retlist