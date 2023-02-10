from numba import njit

@njit
def zscore_norm(data):
    return (data-data.mean())/data.std()

@njit
def minmax_norm(data):
    return (data-data.min())/(data.max()-data.min())