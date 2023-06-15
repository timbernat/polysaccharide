import numpy as np

# typing and subclassing
from typing import Any, Union

# Cheminformatics
from rdkit import Chem


# Type aliases
## General
Numeric = Union[int, float, complex, np.number]
ArrayLike = Union[list, tuple, np.ndarray, np.typing.ArrayLike] # exclude dicts and sets, as they introduce complicated behaviors
JSONSerializable = Union[str, bool, int, float, tuple, list, dict] 

## RDKit
RDMol  = Chem.rdchem.Mol
RDAtom = Chem.rdchem.Atom
RDBond = Chem.rdchem.Bond

## Monomer representation (for structure maping and charging)
SubstructSummary = tuple[str, list[int], bool]  # RDKit graph-match-specific type alias
ResidueSmarts = dict[str, str] # monomer SMARTS strings keyed by residue name
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