# Generic Imports
import re
from datetime import datetime

from itertools import count, islice, product as cartesian_product
from functools import reduce
from collections import deque

from operator import mul
from copy import deepcopy
from pathlib import Path

# Typing and Subclassing
from typing import Any, Callable, Generator, Iterable, Optional, TypeVar, Union
from dataclasses import dataclass

# Units
from pint import Quantity as PintQuantity
from openmm.unit.quantity import Quantity as OMMQuantity


# Math
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

def product(container : Iterable):
    '''Analogous to builtin sum()'''
    return reduce(mul, container)

def int_complement(vals : Iterable[int], bounded : bool=False) -> Generator[int, None, None]:
    '''Generate ordered non-negative integers which don't appear in a collection of integers'''
    _max = max(vals) # cache maximum
    for i in range(_max):
        if i not in vals:
            yield i

    if not bounded: # keep counting past max if unbounded
        yield from count(start=_max + 1, step=1)

@dataclass
class Accumulator:
    '''Compact container for accumulating averages'''
    sum : float = 0.0
    count : int = 0

    @property
    def average(self) -> float:
        return self.sum / self.count


# Functional modifiers and decorators
def generate_repr(cls : Any) -> Any:
    '''Class decorator for autogenerating __repr__ for attributes specified in DISP_ATTRS class attr
    The class this is applied to MUST have implemented an iterable DISP_ATTRS class attribute'''
    assert(hasattr(cls, 'DISP_ATTRS'))
    disp_attrs : Iterable[str] = cls.DISP_ATTRS

    def _repr_generic_(self) -> str:
        attr_str = ', '.join(f'{attr}={getattr(self, attr)}' for attr in disp_attrs)
        return f'{cls.__name__}({attr_str})'
    cls.__repr__ : Callable[[Any], str] = _repr_generic_

    return cls

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

def asiterable(arg_val : Union[Any, Iterable[Any]]) -> Iterable[Any]:
	'''Permits functions expecting iterable arguments to accept singular values'''
	if not isinstance(arg_val, Iterable):
		arg_val = (arg_val,) # turn into single-item tuple (better for memory)
	return arg_val

def aspath(path : Union[Path, str]) -> Path:
	'''Allow functions which expect Paths to also accept strings'''
	if not isinstance(path, Path):
		path = Path(path)
	return path

def asstrpath(strpath : Union[str, Path]) -> str:
	'''Allow functions which expect strings paths to also accept Paths'''
	if not isinstance(strpath, str):
		strpath = str(strpath)
	return strpath


# Tools for iteration 
T = TypeVar('T')
def sliding_window(items : Iterable[T], n : int=1) -> Generator[tuple[T], None, None]:
    '''Generates sliding windows of width n over an iterable collection of items
    E.g. : sliding_window('ABCDE', 3) --> (A, B, C), (B, C, D), (C, D, E)
    '''
    it = iter(items)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

def swappable_loop_order(iter1 : Iterable, iter2 : Iterable, swap : bool=False) -> Iterable[tuple[Any, Any]]:
    '''Enables dynamic swapping of the order of execution of a 2-nested for loop'''
    order = [iter1, iter2] if not swap else [iter2, iter1]
    for pair in cartesian_product(*order):
        yield pair[::(-1)**swap] # reverse order of pair (preserves argument identity)

def progress_iter(itera : Iterable, key : Callable[[Any], str]=lambda x : x) -> Iterable[tuple[str, Any]]:
    '''Iterate through'''
    N = len(itera) # TODO : extend this to work for generators / consumables
    for i, item in enumerate(itera):
        yield (f'{key(item)} ({i + 1} / {N})', item) # +1 converts to more human-readable 1-index for step count


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