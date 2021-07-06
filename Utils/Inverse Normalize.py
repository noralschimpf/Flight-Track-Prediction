import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tqdm

def inv_norm(path_csv: str, path_data: str):

    # Create MinMax Scaler using overall parameters
    lat_scaler = MinMaxScaler(feature_range=[0,1])
    lon_scaler = MinMaxScaler(feature_range=[0,1])
    alt_scaler = MinMaxScaler(feature_range=[0,1])
    vil_scaler = MinMaxScaler(feature_range=[0,1])
    tmp_scaler = MinMaxScaler(feature_range=[0,1])
    uw_scaler = MinMaxScaler(feature_range=[0,1])
    vw_scaler = MinMaxScaler(feature_range=[0,1])

    # lat_scaler.fit(nda_minmax[0,:].reshape(-1,1))
    # lon_scaler.fit(nda_minmax[1,:].reshape(-1,1))
    # alt_scaler.fit(nda_minmax[2,:].reshape(-1,1))
    lat_scaler.fit(np.array([[24.],[50.]]))
    lon_scaler.fit(np.array([[-126.],[-66.]]))
    alt_scaler.fit(np.array([[-1000.],[64000.]]))
    #alt_scaler.fit(np.array([[-1000.],[80000.]]))
    vil_scaler.fit(np.array([[-.00244140625], [80]]))
    tmp_scaler.fit(np.array([[150],[350]]))
    uw_scaler.fit(np.array([[-150],[150]]))
    vw_scaler.fit(np.array([[-150],[150]]))

    '''
    # Generate Transforms from MinMax.CSV
    nda_minmaxes = np.genfromtxt(path_csv,delimiter=',').reshape(3,-1)
    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()
    alt_scaler = MinMaxScaler()
    lat_min, lat_max = nda_minmaxes[0,:].min(), nda_minmaxes[0,:].max()
    lon_min, lon_max = nda_minmaxes[1, :].min(), nda_minmaxes[1, :].max()
    alt_min, alt_max = nda_minmaxes[2, :].min(), nda_minmaxes[2, :].max()

    # Correct Norm/Denorm fit
    lat_scaler.fit(nda_minmaxes[0,:].reshape(-1,1))
    lon_scaler.fit(nda_minmaxes[1,:].reshape(-1,1))
    alt_scaler.fit(nda_minmaxes[2,:].reshape(-1,1))
    '''

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
        nda_data_denormed[:, 3] = lat_scaler.inverse_transform(nda_data[:, 3].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 6] = lat_scaler.inverse_transform(nda_data[:, 6].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 1] = lon_scaler.inverse_transform(nda_data[:, 1].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 4] = lon_scaler.inverse_transform(nda_data[:, 4].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 7] = lon_scaler.inverse_transform(nda_data[:, 7].reshape(-1,1)).reshape(-1)
        nda_data_denormed[:, 2] = alt_scaler.inverse_transform(nda_data[:, 2].reshape(-1, 1)).reshape(-1)
        nda_data_denormed[:, 5] = alt_scaler.inverse_transform(nda_data[:, 5].reshape(-1, 1)).reshape(-1)
        nda_data_denormed[:, 8] = alt_scaler.inverse_transform(nda_data[:, 8].reshape(-1, 1)).reshape(-1)

        df_data_denormed = pd.DataFrame(
            data={'flight plan LAT': nda_data_denormed[:,0], 'flight plan LON': nda_data_denormed[:,1],
                            'flight plan ALT': nda_data_denormed[:,2],
                  'predicted LAT': nda_data_denormed[:,3], 'predicted LON': nda_data_denormed[:,4],
                            'predicted ALT': nda_data_denormed[:,5],
                  'actual LAT': nda_data_denormed[:,6], 'actual LON': nda_data_denormed[:,7],
                            'actual ALT': nda_data_denormed[:,8]})
        df_data_denormed.to_csv('{}/Denormed/{}'.format(path_data,file))

def main():
    valid_products = ['ECHO_TOP', 'VIL', 'tmp', 'uwind', 'vwind']
    mdl_product_dirs = [os.path.join(os.path.abspath('../'), 'Output/{}'.format(x)) for x in os.listdir('../Output') if
                        os.path.isdir('../Output/{}'.format(x)) and any([y in x for y in valid_products])]
    folders_to_denorm = [os.path.join(x, y) for x in mdl_product_dirs for y in os.listdir(x) if 'EPOCHS' in y]
    #folders_to_denorm = [x for x in os.listdir('../Output') if os.path.isdir('../Output/{}'.format(x)) and 'EPOCHS' in x]
    for folder in folders_to_denorm:
        print(folder)
        inv_norm('Data_MinMax.csv',folder)

if __name__ == '__main__':
    main()