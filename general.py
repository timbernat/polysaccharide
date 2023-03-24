# General imports
from datetime import datetime
DATETIME_FMT = '%m-%d-%Y_at_%H-%M-%S_%p' # formatted string which can be used in file names without error

from functools import reduce
from operator import mul

# Typing and Subclassing
from typing import Any, Iterable, Union

# Units
from pint import Quantity as PintQuantity
from openmm.unit.quantity import Quantity as OMMQuantity


def product(container : Iterable):
    '''Analogous to builtin sum()'''
    return reduce(mul, container)

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

def timestamp_now(fmt_str : str=DATETIME_FMT) -> str:
    '''
    Return a string timestamped with the current date and time (at the time of calling)
    Is formatted such that the resulting string can be safely used in a filename
    '''
    return datetime.now().strftime(fmt_str)

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