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
class ChargeMismatchError(Exception):
    '''Specialized error for trying to merge MonomerInfo objects with mismatched charging status'''
    pass

@dataclass
class MonomerInfo(JSONifiable):
    '''For representing monomer information according to the monomer specification <eventual citation>'''
    monomers  : ResidueSmarts
    charges   : ResidueChargeMap = field(default_factory=dict)

    @property
    def SMARTS(self) -> ResidueSmarts:
        return self.monomers # alias of legacy name for convenience
    
    @property
    def has_charges(self) -> bool:
        return self.charges is not None

    def serialize_json_dict(unser_jdict : dict[Any, Union[ResidueSmarts, ResidueChargeMap]]) -> dict[str, JSONSerializable]:
        '''For converting selfs __dict__ data into a form that can be serialized to JSON'''
        ser_jdict = {**unser_jdict}
        # ser_jdict = {} 
        # for key, value in unser_jdict.items():
        #     if (key == 'charges') and (value is None): # skip recording charge entry to file if not set (i.e. if expliclt None)...
        #         continue # ...this is needed as explicit NoneType breaks charge assignment in graph match (expects dict, even if empty)
        #     else:
        #         ser_jdict[key] = value

        return ser_jdict # no special serialization needed (writing to JSON already handles numeric-to-str conversion)
    
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
    
    def __add__(self, other : 'MonomerInfo') -> 'MonomerInfo':
        '''Content-aware method of merging multiple sets of monomer info via the addition operator'''
        if not isinstance(other, MonomerInfo):
            raise NotImplementedError(f'Can only merge {self.__class__.__name__} with another {self.__class__.__name__}, not object of type {type(other)}')
            
        try: # attempt full merge
            return MonomerInfo(
                monomers={**self.monomers, **other.monomers},
                charges={**self.charges, **other.charges}
            )
        except TypeError: # attempt partial merge if either object does not have charges
            return MonomerInfo(
                monomers={**self.monomers, **other.monomers}
            )

    #     if not (self.has_charges or other.has_charges): # neither object has charges
    #         return MonomerInfo(
    #             monomers={**self.monomers, **other.monomers}
    #         )
    #     elif self.has_charges and other.has_charges: # both objects have charges
    #         return MonomerInfo(
    #             monomers={**self.monomers, **other.monomers},
    #             charges={**self.charges, **other.charges}
    #         )
    #     else: # charge status mismatched between objects
    #         raise ChargeMismatchError

    __radd__ = __add__ # support reverse addition

    ## TODO : implement content-aware merge method