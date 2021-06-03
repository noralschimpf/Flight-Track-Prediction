import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from netCDF4 import Dataset as DSet
import pandas as pd
import numpy as np
from typing import List
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir, abspath_fp: str, abspath_ft: str, list_abspath_wc: List[str], list_products: List[str], transform=None, device='cpu'):
        self.device = device
        self.root_dir = root_dir
        self.transform = transform
        # for i in common_dates, fp/common_dates.__contains__(flight) and ..... then abspath for each data item
        self.common_dates = []
        self.flight_plan = abspath_fp
        self.flight_track = abspath_ft
        self.weather_cube = list_abspath_wc
        self.products = list_products

    def __len__(self):
        return len(self.flight_plan)

    def __getitem__(self, idx):
        fp = pd.read_csv(self.flight_plan[idx], usecols=(1, 2, 3)).values
        ft = pd.read_csv(self.flight_track[idx], usecols=(1, 2, 3)).values
        wc = []
        for i in range(len(self.weather_cube[0])):
            wc.append(DSet(self.weather_cube[idx][i], 'r', format='netCDF4'))

        if self.transform:
            fp = self.transform(fp, device=self.device)
            ft = self.transform(ft, device=self.device)
            wc_data = []
            for i in range(len(self.weather_cube[0])):
                for j in range(len(self.products)):
                    if self.products[j] in wc[i].variables.keys():
                        wc_data.append(self.transform(wc[i][self.products[j]][:], device=self.device))

        maxlen = min(len(fp), len(ft), min([len(x) for x in wc_data]))
        wc_data, fp, ft = [x[:maxlen] for x in wc_data], fp[:maxlen], ft[:maxlen]

        return fp, ft, wc_data

    def get_flightname(self, idx):
        splitpath = os.path.normpath(self.flight_plan[idx]).split(os.sep)
        flight_desc = '_'.join(splitpath[-2:])
        return flight_desc


class ToTensor(object):
    def __call__(self, sample, device: str):
        if isinstance(sample, np.ndarray):
            return torch.tensor(sample, device=device, dtype=torch.float32)
        else:
            print("WARNING: object of type " + str(type(sample)) + " not converted")
            return sample

def ValidFiles(root_dir: str, products: List[str], under_min: int = 100):
    dates_fp = os.listdir(root_dir + '/Flight Plans')
    dates_ft = os.listdir(root_dir + '/Flight Tracks')

    dates_wc, wc_to_remove = [], []
    for grp in os.listdir(root_dir + '/Weather Cubes'):
        dates_wc.append(os.listdir(root_dir + '/Weather Cubes/' + grp))
    for g in range(len(dates_wc)):
        for i in range(len(dates_wc[0])):
            if g != 0 and len(dates_wc) != 1:
                if not dates_wc[0][i] in dates_wc[g]:
                    wc_to_remove.append(i)
    wc_to_remove = list(set(wc_to_remove))
    wc_to_remove.sort()
    for i in wc_to_remove:
        dates_wc[0].pop(i)
    dates_wc = dates_wc[0]


    # weather_cube: list(list()) (flight(product))
    # all others: list() (flight)
    common_dates, flight_plan, flight_track, weather_cube, unusable = [], [], [], [], []

    for date in dates_fp:
        if date in dates_ft and date in dates_wc:
            common_dates.append(date)

    for date in common_dates:
        fp_dir = root_dir + '/Flight Plans/' + date
        ft_dir = root_dir + '/Flight Tracks/' + date
        wc_dir = [root_dir + '/Weather Cubes/' + x + '/' + date for x in os.listdir(root_dir + '/Weather Cubes')]
        tmp_fp = os.listdir(fp_dir)
        tmp_ft = os.listdir(ft_dir)
        tmp_wc = [os.listdir(x) for x in wc_dir]

        for fp in tmp_fp:
            flight_desc = '_'.join(fp.split('_')[2:])
            ft = 'Flight_Track_' + flight_desc
            wc = fp.replace('.txt', '.nc')
            try:
                fp_idx = tmp_fp.index(fp)
                ft_idx = tmp_ft.index(ft)
                wc_idx = [tmp_wc[x].index(wc) for x in range(len(tmp_wc))]
                flight_plan.append(os.path.abspath(fp_dir + '/' + fp))
                flight_track.append(os.path.abspath(ft_dir + '/' + ft))
                weather_cube.append([os.path.abspath(x + '/' + wc) for x in wc_dir])

            # Unusables: missing flight plan, flight track, or weather cube file
            except ValueError:
                unusable.append('/'.join([date, flight_desc]))
    print("{} Available Flights, {}  incompatible".format(len(flight_plan), len(unusable)))

    list_underMin = []
    for i in tqdm.trange(len(flight_plan)):
        df_fp = pd.read_csv(flight_plan[i], usecols=(0, 1, 2))
        df_ft = pd.read_csv(flight_track[i], usecols=(0, 1, 2))
        wCubes = [DSet(weather_cube[i][x], 'r', format='netCDF4') for x in range(len(weather_cube[0]))]
        if df_fp.shape[0] < under_min or df_ft.shape[0] < under_min or \
                min([wCubes[x][p].shape[0] for x in range(len(wCubes)) for p in products if p in wCubes[x].variables.keys()]) < under_min:
            list_underMin.append(i)
    print('{} Valid items under minimum entries ({}): {}'.format(len(list_underMin), under_min, list_underMin))
    for i in range(len(list_underMin)):
        flight_desc = flight_plan[list_underMin[i] - i].split('\\')[-2:]
        flight_desc[-1] = '_'.join(flight_desc[-1].split('_')[2:])
        flight_desc = '/'.join(flight_desc)
        unusable.append(flight_desc)
        flight_plan.pop(list_underMin[i] - i)
        flight_track.pop(list_underMin[i] - i)
        weather_cube.pop(list_underMin[i] - i)
    print('{} Available flights, {} unusable flights'.format(len(flight_plan), len(unusable)))

    return flight_plan, flight_track, weather_cube, common_dates, unusable

def SplitStrList(str_list: str, test_idx: int):
    test_idx.sort(reverse=True)
    train_list, test_list = [], []
    for i in range(len(str_list)):
        if i in test_idx: test_list.append(str_list[i])
        else: train_list.append(str_list[i])
    return train_list, test_list


def pad_batch(batch):
    fp = [item[0] for item in batch]
    ft = [item[1] for item in batch]
    wc = [item[2] for item in batch]
    prdlen = len(wc[0])
    batchsize = len(wc)

    #flatten wc and
    wc = [x for item in batch for x in item[2]]

    #expand CIWS wc's to fit cube shapes
    # Fit to Cube height WARNING HEIGH MUST BE ODD
    heights = [x.shape[2] for x in wc]
    maxcenter = int((max(heights)-1)/2)
    if not max(heights) == min(heights):
        wc = [x if x.shape[2]== max(heights) else torch.cat((torch.repeat_interleave(torch.zeros_like(x), maxcenter, dim=2),
                         x,torch.repeat_interleave(torch.zeros_like(x),maxcenter,dim=2)), 2) for x in wc]
    lookahead = wc[0].shape[-4]
    cube_height = wc[0].shape[-3]
    cube_width = wc[0].shape[-2]
    #pad fp, ft, wc
    fp, ft, wc = pad_sequence(fp), pad_sequence(ft), pad_sequence(wc)

    #re-shape wc
    #wc = [wc[x:x + prdlen] for x in range(0, len(wc), prdlen)]
    return [fp, ft, wc.view(-1,batchsize,prdlen,lookahead,cube_height,cube_width,cube_width)]