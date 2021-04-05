import numpy as np

ft_per_nmi = 6076.11549

# Distance Error Reports: Mean Average  Pointwise Error: HORIZONTAL, Along-Track, Cross-Track, VERTICAL
# Pointwise Vertical Error
def PointwiseError(true: np.array, test: np.array, h_units = 'nmi'):
    assert true.shape == test.shape
    assert true.shape[1] == 3
    PVE, PHE = np.zeros((len(true))), np.zeros((len(true)))
    for i in range(len(true)):
        PVE[i] = true[i,2] - test[i,2]
        PHE[i] = dist_between_coords(true[i,:2], test[i,:2], unit=h_units)
    return (PHE, PVE)


# Haversine (as-the-crow-flies) distance between two coordinates
# returned in nautical miles
def dist_between_coords(dest, orig, unit = 'nmi', rEarth = 6370997):
    divisor = {'m': 1, 'km': 1000, 'nmi': 1852}
    lat2_rad, lat1_rad, delta_lons_rad = np.radians([dest[0], orig[0], dest[1]-orig[1]])
    a = np.sin((lat2_rad - lat1_rad) / 2) ** 2 + (np.cos(lat1_rad)*np.cos(lat2_rad)*(np.sin(delta_lons_rad/2) ** 2))
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return rEarth * c / divisor[unit]


def L2Norm(err: np.array):
    norm = 0
    assert len(err.shape) == 2
    for i in range(err.shape[0]):
        dist  = 0
        for j in range(err.shape[1]):
            dist += abs((err[i,j]))**2
        norm += np.sqrt(dist)
    return np.sqrt(norm)


def MSE(act: np.array, test:np.array):
    assert act.shape != test.shape
    mse = 0
    # for each point
    for i in range(act.shape[0]):
        # for each dimension
        dist = 0
        for j in range(act.shape[1]):
            dist += abs((act[i,j] - test[i,j]))**2
        mse += np.sqrt(dist)
    return mse/act.shape[0]


def reduction(l2_orig, l2_new):
    if isinstance(l2_orig, np.ndarray) or isinstance(l2_orig, list):
        if len(l2_orig) >1:
            return (np.var(l2_orig) - np.var(l2_new))/np.var(l2_orig)
    else:
        return (l2_orig - l2_new)/l2_orig