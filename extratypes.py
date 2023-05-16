from typing import Any, Iterable, Union
from numpy.typing import ArrayLike
from numpy import number, ndarray

# Common matplotlib types
from matplotlib.pyplot import Figure, Axes # used to avoid matplotlib import in downstream modules if only needed for typing
from matplotlib.colors import Colormap, Normalize

from rdkit import Chem


# Type aliases
## General
Numeric = Union[int, float, complex, number]
ArrayLike = Union[list, tuple, ndarray, ArrayLike] # exclude dicts and sets, as they introduce complicated behaviors
JSONSerializable = Union[str, bool, int, float, tuple, list, dict] 

## OpenFF / graph match
SubstructSummary = tuple[str, list[int], bool]  # RDKit graph-match-specific type alias

## RDKit
RDMol  = Chem.rdchem.Mol
RDAtom = Chem.rdchem.Atom
RDBond = Chem.rdchem.Bond

## Charge mapping and averaging - TODO : remove these, as they have been migrated over to .charging
AtomIDMap = dict[str, dict[int, tuple[int, str]]]
ChargeMap = dict[int, float] 
ResidueChargeMap = dict[str, ChargeMap]

def asiterable(arg_val : Union[Any, Iterable[Any]]) -> Iterable[Any]:
	'''Permits functions expecting iterable arguments to accept singular values'''
	if not isinstance(arg_val, Iterable):
		arg_val = tuple(arg_val)
	return arg_val

# Typechecking functions
def isnumeric(var: Any) -> bool:
	'''Check if a variable is numerical'''
	return isinstance(var, Numeric.__args__)

def isarraylike(var: Any) -> bool:
	'''Check if a variable is numerical'''
	return isinstance(var, ArrayLike.__args__)