# Generic Imports
import re
from datetime import datetime

from functools import reduce
from operator import mul
from copy import deepcopy

# Typing and Subclassing
from typing import Any, Callable, Iterable, Optional, Union
from dataclasses import dataclass

# Units
from pint import Quantity as PintQuantity
from openmm.unit.quantity import Quantity as OMMQuantity


# Greek Characters
greek_letter_names = [ # names for greek character literals
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
    'rho', 'sigma_end', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
]

_greek_start_idxs = { # indices where each case of the Greek alphabet starts in Unicode
    'LOWER' : 945,
    'UPPER' : 913
}

for case, idx in _greek_start_idxs.items():
    globals()[f'GREEK_{case}'] = { # add dicts to global namespace
        letter_name : chr(idx + i)
            for i, letter_name in enumerate(greek_letter_names)
    }

# Math
def product(container : Iterable):
    '''Analogous to builtin sum()'''
    return reduce(mul, container)

@dataclass
class Accumulator:
    '''Compact container for accumulating averages'''
    sum : float = 0.0
    count : int = 0

    @property
    def average(self) -> float:
        return self.sum / self.count

# Functional operations and decorators
def optional_in_place(funct : Callable[[Any, Any], None]) -> Callable[[Any, Any], Optional[Any]]:
    '''Decorator function for allowing in-place (writeable) functions which modify object attributes
    to be not performed in-place (i.e. read-only), specified by a boolean flag'''
    def in_place_wrapper(obj : Any, *args, in_place : bool=False, **kwargs) -> Optional[Any]: # read-only by default
        '''If not in-place, create a clone on which the method is executed'''
        if in_place:
            funct(obj, *args, **kwargs) # default call to writeable method - implicitly returns None
        else:
            copy_obj = deepcopy(obj) # clone object to avoid modifying original
            funct(copy_obj, *args, **kwargs) 
            return copy_obj # return the new object
    return in_place_wrapper


# Data containers / data structures
@optional_in_place
def modify_dict(path_dict : dict[Any, Any], modifier_fn : Callable[[Any, Any], tuple[Any, bool]]) -> None:
    '''Recursively modifies all values in a dict in-place according to some function'''
    for key, val in path_dict.items():
        if isinstance(val, dict): # recursive call if sub-values are also dicts with Paths
            modify_dict(val, modifier_fn)
        else:
            path_dict[key] = modifier_fn(key, val) 
    
def iter_len(itera : Iterable):
    '''
    Get size of an iterable object where ordinary len() call is invalid (namely a generator)
    Note that this will "use up" a generator upon iteration
    '''
    return sum(1 for _  in itera)

def sort_dict_by_values(targ_dict : dict, reverse : bool=False) -> dict[Any, Any]:
    '''Sort a dictionary according to the values of each key'''
    return { # sort dict in ascending order by size
        key : targ_dict[key]
            for key in sorted(targ_dict, key=lambda k : targ_dict[k], reverse=reverse)
    }

# Unit handling
class MissingUnitsError(Exception):
    pass

def hasunits(obj : Any) -> bool:
    '''Naive but effective way of checking for pint and openmm units'''
    return any(hasattr(obj, attr) for attr in ('unit', 'units')) 

def strip_units(coords : Union[tuple, PintQuantity, OMMQuantity]) -> tuple[float]:
    '''
    Sanitize coordinate tuples for cases which require unitless quantities
    Specifically needed since OpenMM and pint each have their own Quantity and Units classes
    '''
    if isinstance(coords, PintQuantity):
        return coords.magnitude
    elif isinstance(coords, OMMQuantity):
        return coords._value

    return coords

# Date / time formatting
@dataclass
class Timestamp:
    '''For storing information on date processing'''
    fmt_str : str = '%m-%d-%Y_at_%H-%M-%S_%p'# should be formatted such that the resulting string can be safely used in a filename (i.e. no slashes)
    regex : Union[str, re.Pattern] = re.compile(r'\d{2}-\d{2}-\d{4}_at_\d{2}-\d{2}-\d{2}_\w{2}')

    def timestamp_now(self) -> str:
        '''Return a string timestamped with the current date and time (at the time of calling)'''
        return datetime.now().strftime(self.fmt_str)

    def extract_datetime(self, timestr : str) -> datetime:
        '''De-format a string containing a timestamp and extract just the timestamp as a datetime object'''
        timestamps = re.search(self.regex, timestr) # pull out JUST the datetime formatting component
        return datetime.strptime(timestamps.group(), self.fmt_str) # convert to datetime object