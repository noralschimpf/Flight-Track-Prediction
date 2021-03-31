import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tqdm

def inv_norm(path_csv: str, path_data: str):
    # Generate Transforms from MinMax.CSV
    nda_minmaxes = np.genfromtxt(path_csv,delimiter=',').reshape(3,-1)

    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()
    alt_scaler = MinMaxScaler()

    lat_min, lat_max = nda_minmaxes[0,:].min(), nda_minmaxes[0,:].max()
    lon_min, lon_max = nda_minmaxes[1, :].min(), nda_minmaxes[1, :].max()
    alt_min, alt_max = nda_minmaxes[2, :].min(), nda_minmaxes[2, :].max()
    '''DEPRECATED
    lat_scaler.fit(np.array([lat_min, lat_max]).reshape(-1,1))
    lon_scaler.fit(np.array([lon_min, lon_max]).reshape(-1,1))
    alt_scaler.fit(np.array([alt_min, alt_max]).reshape(-1,1))'''

    # Correct Norm/Denorm fit
    lat_scaler.fit(nda_minmaxes[0,:].reshape(-1,1))
    lon_scaler.fit(nda_minmaxes[1,:].reshape(-1,1))
    alt_scaler.fit(nda_minmaxes[2,:].reshape(-1,1))

    # Inverse Transform each Saved CSV
    if not os.path.isdir('{}/Denormed'.format(path_data)):
        os.mkdir('{}/Denormed'.format(path_data))
    files = [x for x in os.listdir(path_data) if os.path.isfile(os.path.join(path_data,x))]

    for file in tqdm.tqdm(files):
        path_file = os.path.join(path_data, file)
        df_data = pd.read_csv(path_file)
        nda_data = df_data.values[:,1:]

        nda_data_denormed = np.zeros_like(nda_data)
        nda_data_denormed[:, 0] = lat_scaler.inverse_transform(nda_data[:, 0].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 2] = lat_scaler.inverse_transform(nda_data[:, 2].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 4] = lat_scaler.inverse_transform(nda_data[:, 4].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 1] = lon_scaler.inverse_transform(nda_data[:, 1].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 3] = lon_scaler.inverse_transform(nda_data[:, 3].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 5] = lon_scaler.inverse_transform(nda_data[:, 5].reshape(-1,1)).reshape(-1)

        df_data_denormed = pd.DataFrame(
            data={'flight plan LAT': nda_data_denormed[:,0], 'flight plan LON': nda_data_denormed[:,1],
                  'predicted LAT': nda_data_denormed[:,2], 'predicted LON': nda_data_denormed[:,3],
                  'actual LAT': nda_data_denormed[:,4], 'actual LON': nda_data_denormed[:,5]})
        df_data_denormed.to_csv('{}/Denormed/{}'.format(path_data,file))

def main():
    folders_to_denorm = [x for x in os.listdir('../Output') if os.path.isdir('../Output/{}'.format(x)) and 'EPOCHS' in x]
    for folder in folders_to_denorm:
        print(folder)
        inv_norm('Data_MinMax.csv','../Output/{}'.format(folder))

if __name__ == '__main__':
    main()