# General imports
from datetime import datetime

from functools import reduce
from operator import mul

# Typing and Subclassing
from typing import Any, Callable, Iterable, Optional, Union

# Units
from pint import Quantity as PintQuantity
from openmm.unit.quantity import Quantity as OMMQuantity


# Greek Characters
greek_letter_names = [ # names for greek character literals
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
    'rho', 'sigma_end', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
]

greek_start_idxs = { # indices where each case of the Greek alphabet starts in Unicode
    'LOWER' : 945,
    'UPPER' : 913
}

for case, idx in greek_start_idxs.items():
    globals()[f'GREEK_{case}'] = { # add dicts to global namespace
        letter_name : chr(idx + i)
            for i, letter_name in enumerate(greek_letter_names)
    }

# Missing should-be builtins
def product(container : Iterable):
    '''Analogous to builtin sum()'''
    return reduce(mul, container)

# Data containers
def _modify_dict(path_dict : dict[Any, Any], modifier_fn : Callable[[Any, Any], tuple[Any, bool]]) -> None:
    '''Recursively modifies all values in a dict in-place according to some function'''
    for key, val in path_dict.items():
        if isinstance(val, dict): # recursive call if sub-values are also dicts with Paths
            modify_dict(val, modifier_fn)
        else:
            path_dict[key] = modifier_fn(key, val) 
    
def modify_dict(path_dict : dict[Any, Any], modifier_fn : Callable[[Any, Any], Any], in_place : bool=False) -> Optional[dict[Any, Any]]:
    '''Recursively modifies all Path-like values in a dict according to some function
    Can specify whether to modify in-place or return a modified copy to avoid overwrites'''
    if in_place:
        _modify_dict(path_dict, modifier_fn=modifier_fn) # implcitly returns None
    else:
        copy_dict = {k : v for k, v in path_dict.items()} # create a copy to avoid overwrites
        _modify_dict(copy_dict, modifier_fn=modifier_fn) # modify the copy in-place
        return copy_dict

# Helper methods for builtin data structures
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

# Date and time formatting
DATETIME_FMT = '%m-%d-%Y_at_%H-%M-%S_%p' # formatted string which can be used in file names without error

def extract_time(timestamp : str) -> str:
    '''De-format a timestamped string and extract just the timestamp'''
    return datetime.strptime(timestamp, DATETIME_FMT)

def timestamp_now(fmt_str : str=DATETIME_FMT) -> str:
    '''
    Return a string timestamped with the current date and time (at the time of calling)
    Is formatted such that the resulting string can be safely used in a filename
    '''
    return datetime.now().strftime(fmt_str)

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