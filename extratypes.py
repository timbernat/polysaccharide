import numpy as np

# typing and subclassing
from .filetree import JSONifiable
from typing import Any, Iterable, Union
from dataclasses import dataclass, field

# Common matplotlib types
from matplotlib.pyplot import Figure, Axes # used to avoid matplotlib import in downstream modules if only needed for typing
from matplotlib.colors import Colormap, Normalize

# Cheminformatics
from rdkit import Chem


# Type aliases
## General
Numeric = Union[int, float, complex, np.number]
ArrayLike = Union[list, tuple, np.ndarray, np.typing.ArrayLike] # exclude dicts and sets, as they introduce complicated behaviors
JSONSerializable = Union[str, bool, int, float, tuple, list, dict] 

## OpenFF / graph match
SubstructSummary = tuple[str, list[int], bool]  # RDKit graph-match-specific type alias

## RDKit
RDMol  = Chem.rdchem.Mol
RDAtom = Chem.rdchem.Atom
RDBond = Chem.rdchem.Bond

## Monomer representation (for structure maping and charging)
ResidueSmarts = dict[str, str] # monomer SMARTS strings keyed by residue name
AtomIDMap = dict[str, dict[int, tuple[int, str]]] 
ChargeMap = dict[int, float] 
ResidueChargeMap = dict[str, ChargeMap]


# Typing functions
def asiterable(arg_val : Union[Any, Iterable[Any]]) -> Iterable[Any]:
	'''Permits functions expecting iterable arguments to accept singular values'''
	if not isinstance(arg_val, Iterable):
		arg_val = (arg_val,) # turn into single-item tuple (better for memory)
	return arg_val

## typechecking 
def isnumeric(var: Any) -> bool:
	'''Check if a variable is numerical'''
	return isinstance(var, Numeric.__args__)

def isarraylike(var: Any) -> bool:
	'''Check if a variable is numerical'''
	return isinstance(var, ArrayLike.__args__)


# Custom classes / types
@dataclass
class MonomerInfo(JSONifiable):
    '''For representing monomer information according to the monomer specification <eventual citation>'''
    monomers  : ResidueSmarts
    charges   : ResidueChargeMap = field(default=None)

    @property
    def SMARTS(self) -> ResidueSmarts:
        return self.monomers # alias of legacy name for convenience
    
    @property
    def has_charges(self) -> bool:
        return self.charges is not None

    def serialize_json_dict(unser_jdict : dict[Any, Union[ResidueSmarts, ResidueChargeMap]]) -> dict[str, JSONSerializable]:
        '''For converting selfs __dict__ data into a form that can be serialized to JSON'''
        return unser_jdict # no special serialization needed (writing to JSON already handles numeric-to-str conversion)
    
    def unserialize_json_dict(ser_jdict : dict[str, JSONSerializable]) -> dict[Any, Union[ResidueSmarts, ResidueChargeMap]]:
        '''For unserializing charged residue maps in charged monomer JSON files'''
        unser_jdict = {}
        for key, value in ser_jdict.items():
            try: # TOSELF : need try-except instead of explicit check for "charges" key since object hook is applied to ALL subdicts (not just main)
                unser_jdict[key] = { # convert string-keyed indices and charges back to numeric types
                    int(substruct_id) : float(charge)
                        for substruct_id, charge in value.items()
                }
            except (ValueError, AttributeError):
                unser_jdict[key] = value
        
        return unser_jdict
    
    ## TODO : implement content-aware merge method