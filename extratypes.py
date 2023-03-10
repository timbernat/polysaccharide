from typing import Any, Callable, Union, Optional

from numpy.typing import ArrayLike
from numpy import number, ndarray

from pint import Quantity as PintQuantity
from openmm.unit.quantity import Quantity as OMMQuantity



Numeric = Union[int, float, complex, number]
ArrayLike = Union[list, tuple, ndarray, ArrayLike] # exclude dicts and sets, as they introduce complicated behaviors
JSONSerializable = Union[str, bool, int, float, tuple, list, dict] 

def isnumeric(var: Any) -> bool:
  '''Check if a variable is numerical'''
  return isinstance(var, Numeric.__args__)

def isarraylike(var: Any) -> bool:
  '''Check if a variable is numerical'''
  return isinstance(var, ArrayLike.__args__)

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