import numpy as np
import ruptures as rpt


def normalize(series : np.ndarray) -> np.ndarray:
    '''Normalize a series of data to have all values lie in the range [-1, 1]'''
    return (series - series.min()) / (series.max() - series.min())

def standardize(series : np.ndarray) -> np.ndarray:
    '''Standardize a series of data to have mean 0 and standard deviation of 1'''
    return (series - series.mean()) / series.std() 

def autocorrelate(series : np.ndarray) -> np.ndarray:
    '''Compute autocorrelation of a vector series of data points'''
    assert(series.ndim == 1) # only operate for vectors of data
    
    series = standardize(series)
    autocorr = np.correlate(series, series, mode='full') # need full mode to get proper length
    autocorr = autocorr[autocorr.size//2:] # only keep second half of array (Hermitian, so symmetric about midpoint)
    autocorr /= series.size # normalize to range [-1, 1] by including number of datapoints (due to variance estimator)

    return autocorr

def RMSE(pred : np.ndarray, obs : np.ndarray) -> float:
    '''Computes root-mean squared error between predicted and observed sets of values'''
    sq_err = (obs - pred)**2
    return np.sqrt(sq_err.mean())

def equil_loc(series : np.ndarray) -> int:
    '''Estimate the index in a series after which equilibration has occurred'''
    if not isinstance(series, np.ndarray):
        series = np.array(series)

    algo = rpt.Binseg(model='l2').fit(series)
    change_times = algo.predict(n_bkps=1)

    return change_times[0]