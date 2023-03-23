# General imports
from datetime import datetime
DATETIME_FMT = '%m-%d-%Y_at_%H-%M-%S_%p' # formatted string which can be used in file names without error

from functools import reduce
from operator import mul

# Typing and Subclassing
from typing import Any, Iterable


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