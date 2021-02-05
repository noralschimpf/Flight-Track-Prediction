import torch, tqdm
from torch.utils.data import Dataset
from netCDF4 import Dataset as DSet
import pandas as pd
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu'):
        self.device = device
        self.root_dir = root_dir
        self.transform = transform
        # for i in common_dates, fp/common_dates.__contains__(flight) and ..... then abspath for each data item
        self.common_dates = []
        self.flight_plan = []
        self.flight_track = []
        self.weather_cube = []
        self.unusable = []
        dates_fp = os.listdir(root_dir + '/Flight Plans')
        dates_ft = os.listdir(root_dir + '/Flight Tracks')
        dates_wc = os.listdir(root_dir + '/Weather Cubes')

        for date in dates_fp:
            if dates_ft.__contains__(date) and dates_wc.__contains__(date):
                self.common_dates.append(date)

        for date in self.common_dates:
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
                    self.flight_plan.append(os.path.abspath(fp_dir + '/' + fp))
                    self.flight_track.append(os.path.abspath(ft_dir + '/' + ft))
                    self.weather_cube.append(os.path.abspath(wc_dir + '/' + wc))

                # Unusables: missing flight plan, flight track, or weather cube file
                except ValueError:
                    self.unusable.append('/'.join([date, flight_desc]))
        print("{} Available Flights, {}  incompatible".format(len(self.flight_plan),len(self.unusable)))

    def validate_sets(self, underMin: int = 100):
        list_underMin = []
        print('Validating Flight Data:')
        for i in tqdm.trange(len(self.flight_plan)):
            df_fp = pd.read_csv(self.flight_plan[i], usecols=(0,1,2))
            df_ft = pd.read_csv(self.flight_track[i], usecols=(0,1,2))
            wCubes = DSet(self.weather_cube[i], 'r', format='netCDF4')
            if df_fp.shape[0] < underMin or df_ft.shape[0] < underMin or wCubes['Echo_Top'].shape[0] < underMin:
                list_underMin.append(i)
        print('{} Valid items under minimum entries ({}): {}'.format(len(list_underMin), underMin, list_underMin))
        for i in range(len(list_underMin)):
            flight_desc = self.flight_plan[list_underMin[i]-i].split('\\')[-2:]
            flight_desc[-1] = '_'.join(flight_desc[-1].split('_')[2:])
            flight_desc = '/'.join(flight_desc)
            self.unusable.append(flight_desc)
            self.flight_plan.pop(list_underMin[i]-i)
            self.flight_track.pop(list_underMin[i]-i)
            self.weather_cube.pop(list_underMin[i]-i)
        print('{} Available flights, {} unusable flights'.format(len(self.flight_plan), len(self.unusable)))


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
        flight_desc = '_'.join(self.flight_plan[idx].split('\\')[-2:])
        return flight_desc




class ToTensor(object):
    def __call__(self, sample, device: str):
        if isinstance(sample, np.ndarray):
            return torch.tensor(sample, device=device, dtype=torch.float32)
        else:
            print("WARNING: object of type " + str(type(sample)) + " not converted")
            return sample
