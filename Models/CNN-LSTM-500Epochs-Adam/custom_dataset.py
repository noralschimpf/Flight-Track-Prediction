import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from netCDF4 import Dataset as DSet
import pandas as pd
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir, abspath_fp: str, abspath_ft: str, abspath_wc: str, transform=None, device='cpu'):
        self.device = device
        self.root_dir = root_dir
        self.transform = transform
        # for i in common_dates, fp/common_dates.__contains__(flight) and ..... then abspath for each data item
        self.common_dates = []
        self.flight_plan = abspath_fp
        self.flight_track = abspath_ft
        self.weather_cube = abspath_wc

    def __len__(self):
        return len(self.flight_plan)

    def __getitem__(self, idx):
        fp = pd.read_csv(self.flight_plan[idx], usecols=(0, 1, 2)).values
        ft = pd.read_csv(self.flight_track[idx], usecols=(0, 1, 2)).values
        wc = DSet(self.weather_cube[idx], 'r', format='netCDF4')

        if self.transform:
            fp = self.transform(fp, device=self.device)
            ft = self.transform(ft, device=self.device)
            wc = self.transform(wc['Echo_Top'][:], device=self.device)

        maxlen = min(len(fp), len(ft), len(wc))
        wc, fp, ft = wc[:maxlen], fp[:maxlen], ft[:maxlen]

        return fp, ft, wc

    def get_flightname(self, idx):
        flight_desc = '_'.join(self.flight_plan[idx].split('/')[-2:])
        return flight_desc


class ToTensor(object):
    def __call__(self, sample, device: str):
        if isinstance(sample, np.ndarray):
            return torch.tensor(sample, device=device, dtype=torch.float32)
        else:
            print("WARNING: object of type " + str(type(sample)) + " not converted")
            return sample

def ValidFiles(root_dir: str, under_min: int = 100):
    dates_fp = os.listdir(root_dir + '/Flight Plans')
    dates_ft = os.listdir(root_dir + '/Flight Tracks')
    dates_wc = os.listdir(root_dir + '/Weather Cubes')

    common_dates, flight_plan, flight_track, weather_cube, unusable = [], [], [], [], []

    for date in dates_fp:
        if dates_ft.__contains__(date) and dates_wc.__contains__(date):
            common_dates.append(date)

    for date in common_dates:
        fp_dir = root_dir + '/Flight Plans/' + date
        ft_dir = root_dir + '/Flight Tracks/' + date
        wc_dir = root_dir + '/Weather Cubes/' + date
        tmp_fp = os.listdir(fp_dir)
        tmp_ft = os.listdir(ft_dir)
        tmp_wc = os.listdir(wc_dir)

        for fp in tmp_fp:
            flight_desc = '_'.join(fp.split('_')[2:])
            ft = 'Flight_Track_' + flight_desc
            wc = fp.replace('.txt', '.nc')
            try:
                fp_idx = tmp_fp.index(fp)
                ft_idx = tmp_ft.index(ft)
                wc_idx = tmp_wc.index(wc)
                flight_plan.append(os.path.abspath(fp_dir + '/' + fp))
                flight_track.append(os.path.abspath(ft_dir + '/' + ft))
                weather_cube.append(os.path.abspath(wc_dir + '/' + wc))

            # Unusables: missing flight plan, flight track, or weather cube file
            except ValueError:
                unusable.append('/'.join([date, flight_desc]))
    print("{} Available Flights, {}  incompatible".format(len(flight_plan), len(unusable)))

    list_underMin = []
    for i in tqdm.trange(len(flight_plan)):
        df_fp = pd.read_csv(flight_plan[i], usecols=(0, 1, 2))
        df_ft = pd.read_csv(flight_track[i], usecols=(0, 1, 2))
        wCubes = DSet(weather_cube[i], 'r', format='netCDF4')
        if df_fp.shape[0] < under_min or df_ft.shape[0] < under_min or wCubes['Echo_Top'].shape[0] < under_min:
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
    fp, ft, wc = pad_sequence(fp, batch_first=True), pad_sequence(ft, batch_first=True), pad_sequence(wc, batch_first=True)
    return [fp, ft, wc]