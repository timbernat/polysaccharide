from typing import Any, Callable, Union, Optional
from numpy.typing import ArrayLike
from numpy import number, ndarray

# Type aliases
# -- General
Numeric = Union[int, float, complex, number]
ArrayLike = Union[list, tuple, ndarray, ArrayLike] # exclude dicts and sets, as they introduce complicated behaviors
JSONSerializable = Union[str, bool, int, float, tuple, list, dict] 

# -- OpenFF / graph match
SubstructSummary = tuple[str, list[int], bool]  # RDKit graph-match-specific type alias

# -- Charge mapping and averaging - TODO : remove these, as they have been migrated over to .charging
AtomIDMap = dict[str, dict[int, tuple[int, str]]]
ChargeMap = dict[int, float] 
ResidueChargeMap = dict[str, ChargeMap]

# Typechecking functions
def isnumeric(var: Any) -> bool:
  '''Check if a variable is numerical'''
  return isinstance(var, Numeric.__args__)

def isarraylike(var: Any) -> bool:
  '''Check if a variable is numerical'''
  return isinstance(var, ArrayLike.__args__)