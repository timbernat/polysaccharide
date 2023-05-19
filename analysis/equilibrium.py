'''Utilities for assessing equilibration times and equilibrium averages of computed properties'''

# typing and subclassing
from abc import ABC, abstractstaticmethod

# numeric and statistical processing
import numpy as np
import ruptures as rpt
from pymbar import timeseries


class EquilibriumDetector(ABC):
    '''Provides interface for various methods of determining equilibration in a property time series'''
    @abstractstaticmethod
    def equil_loc(series : np.ndarray, *args, **kwargs) -> int:
        '''Returns the index in the time series at which equilibrium is determined to have occurred'''
        pass

class BinSegEquilDetector(EquilibriumDetector):
    '''Detects equilibria using PELT (Pruned Exact Linear Time) Binary Segmentation (as implemented by ruptures)'''
    def equil_loc(series : np.ndarray) -> int:
        '''Estimate the index in a series after which equilibration has occurred'''
        if not isinstance(series, np.ndarray):
            series = np.array(series)

        algo = rpt.Binseg(model='l2').fit(series)
        change_times = algo.predict(n_bkps=1)

        return change_times[0]
    
class PyMBAREquilDetector(EquilibriumDetector):
    '''Detects equilibria using a heuristic that maximizes number of effectively uncorrelated samples (as implemented in pymbar)'''
    def equil_loc(series : np.ndarray, fast : bool=False) -> int:
        t, *_ = timeseries.detect_equilibration(series, fast=fast)
        return t

EQUIL_DETECTOR_REGISTRY = {
    equil_det.__name__ : equil_det
        for equil_det in EquilibriumDetector.__subclasses__()
}