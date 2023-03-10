# General imports
from datetime import datetime
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
    Note that this will "use up" the generator upon iteration
    '''
    return sum(1 for _  in itera)

def sort_dict_by_values(targ_dict : dict, reverse : bool=False) -> dict[Any, Any]:
    '''Sort a dictionary according to the values of each key'''
    return { # sort dict in ascending order by size
        key : targ_dict[key]
            for key in sorted(targ_dict, key=lambda k : targ_dict[k], reverse=reverse)
    }

def timestamp_now() -> str:
    '''Return a timestamp string with the current date and time
    Is formatted such that '''
    return datetime.now().strftime('%m-%d-%Y_at_%H-%M-%S_%p')