'''For calculating information about a polymer chain ab mono (i.e. from monomer information)'''

# logging
import logging
LOGGER = logging.getLogger(__name__)

# Numeric and string processing
import re
import numpy as np

# Typing and subclassing
from typing import Any, Union
from dataclasses import dataclass, field

from .exceptions import InsufficientChainLengthError
from ..filetree import JSONSerializable, JSONifiable
from ..molutils.rdmol.rdbond import hydrogenate_rdmol_ports
from ..molutils.rdmol.rdlabels import clear_atom_map_nums
from ..extratypes import ResidueSmarts, ResidueChargeMap
from .exceptions import ChargeMismatchError

# Cheminformatics
from rdkit import Chem


# Representation class for monomer information
@dataclass
class MonomerInfo(JSONifiable):
    '''For representing monomer information according to the monomer specification <eventual citation>'''
    monomers  : ResidueSmarts    = field(default_factory=dict)
    charges   : ResidueChargeMap = field(default_factory=dict)

    @property
    def SMARTS(self) -> ResidueSmarts:
        return self.monomers # alias of legacy name for convenience
    
    @property
    def has_charges(self) -> bool:
        return bool(self.charges)

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
        cls = self.__class__
        if not isinstance(other, cls):
            raise NotImplementedError(f'Can only merge {cls.__name__} with another {cls.__name__}, not object of type {type(other)}')

        if self.has_charges and other.has_charges: # only write charges to child MonomerInfo if both parents have charges
            return MonomerInfo(
                monomers={**self.monomers, **other.monomers},
                charges={**self.charges, **other.charges}
            )
        else:
            return MonomerInfo(
                monomers={**self.monomers, **other.monomers}
            )

    __radd__ = __add__ # support reverse addition

# Functions for estimating chain parameters from monomer info
def unique_monomers(monomer_smarts : ResidueSmarts) -> set[str]:
    '''Returns set of SMILES strings of completed (i.e. hydrogenated, port-free) monomers present in a collection of monomer SMARTS''' 
    unique_mono = set()
    for SMARTS in monomer_smarts.values():
        rdmol = Chem.MolFromSmarts(SMARTS)
        clear_atom_map_nums(rdmol, in_place=True) 
        hydrogenate_rdmol_ports(rdmol, in_place=True) 
        unique_mono.add(Chem.MolToSmiles(rdmol)) # TODO : eventually make this SMART-based (or even better RDMol-based); can't for now since hydrogenated fragments don't equate

    return unique_mono

def count_ports_in_rdmol(rdmol : Chem.rdchem.Mol) -> int:
    '''Counts the number of ports present in an RDMol monomer'''
    return sum(1 for atom in rdmol.GetAtoms() if not atom.GetAtomicNum())

def count_ports_in_SMARTS(SMARTS : str) -> int:
    '''Counts the number of ports present in a monomer SMARTS'''
    # return SMARTS.count('*') # naive but relatively effective for nominal cases
    return count_ports_in_rdmol(Chem.MolFromSmarts(SMARTS))

is_term_by_resname = lambda res_name : bool(re.search('TERM', res_name, flags=re.IGNORECASE)) # naive and far less general test when explicitly labelled
is_term_by_smarts  = lambda SMARTS : count_ports_in_SMARTS(SMARTS) == 1 # terminal monomers must have exactly 1 port by definition
is_term_by_rdmol   = lambda rdmol : count_ports_in_rdmol(rdmol) == 1

def count_middle_and_term_mono(monomer_smarts : ResidueSmarts) -> tuple[int, int]:
    '''Determine how many of the monomers in a base set are middle vs terminal
    Results return is number of middle monomers, followed by the number of terminal monomers'''
    group_counts = [0, 0]
    for SMARTS in monomer_smarts.values():
        group_counts[is_term_by_smarts(SMARTS)] += 1 # index by bool
    
    n_mid, n_term = group_counts # unpack purely for documentation and statification
    return (n_mid, n_term)

def is_linear(monomer_smarts : ResidueSmarts) -> bool:
    '''Identify if a polymer is a linear, unbranched chain'''
    return not any( # linear polymers have at most 1 bond entering and 1 bond leaving every monomer
        (count_ports_in_SMARTS(SMARTS) > 2)
            for SMARTS in monomer_smarts.values()
    )

def is_homopolymer(monomer_smarts : ResidueSmarts) -> bool:
    '''Identify if a polymer is a homopolymer (i.e. only 1 type of middle monomer)'''
    # n_mid, n_term = count_middle_and_term_mono(monomer_smarts) # TODO : reimplement with comparison of port-hydrogenated monomers
    # return (n_mid == 1)
    return (len(unique_monomers(monomer_smarts)) == 1) # by definition, a homopolymer only has 1 unique class of monomer

def is_linear_homopolymer(monomer_smarts : ResidueSmarts) -> bool:
    '''Identify if a polymer is a linear homopolymer'''
    return is_linear(monomer_smarts) and is_homopolymer(monomer_smarts)

def estimate_chain_len(monomer_smarts : ResidueSmarts, DOP : int) -> int:
    '''Given a set of monomers and the desired degree of polymerization, estimate the length of the resulting chain
    !NOTE! : As-implemented, only works for linear homopolymers and block copolymers with equal an distribution of monomers'''
    # TOSELF : omitted logging for now, as it gets repeated on EVERY cycle in when called estimate_DOP_lower
    num_mono = len(monomer_smarts)

    mono_term    = np.zeros(num_mono, dtype=bool) # terminality of each monomer (i.e. whether or not it is a term group)
    mono_multip  = np.zeros(num_mono, dtype=int) # multiplicity of each polymer (i.e. how many times is occurs in a chain)
    mono_contrib = np.zeros(num_mono, dtype=int) # contribution of each monomer (i.e. how many atoms does it add to the chain)

    for i, (resname, SMARTS) in enumerate(monomer_smarts.items()):
        monomer = Chem.MolFromSmarts(SMARTS)

        num_atoms = monomer.GetNumAtoms()
        num_ports = count_ports_in_rdmol(monomer)
        is_term = is_term_by_rdmol(monomer)

        mono_term[i] = is_term
        mono_multip[i] = is_term # temporarily set middle monomer contribution to 0
        mono_contrib[i] = num_atoms - num_ports

    num_term = sum(mono_term)
    num_mid  = num_mono - num_term # assumed that all monomers are either terminal or not
    mono_multip[~mono_term] = (DOP - num_term) / num_mid # naive assumption that all middle monomers contribute rest of chain equally (for homopolymers, this is always true)

    N = mono_contrib @ mono_multip # compute dot product to yield final count
    
    return N

def estimate_DOP_lower(monomer_smarts : ResidueSmarts, max_chain_len : int, min_DOP : int=3) -> int:
    '''Returns the largest DOP for a set of monomers which yields a chain no longer than the specified chain length'''
    base_chain_len = estimate_chain_len(monomer_smarts, min_DOP)
    if base_chain_len > max_chain_len: # pre-check when optimization is impossible
        raise InsufficientChainLengthError(f'Even shortest possible chain (DOP={min_DOP}, N={base_chain_len}) is longer than the specified max length of {max_chain_len} atoms')

    DOP = min_DOP 
    while estimate_chain_len(monomer_smarts, DOP + 1) < max_chain_len: # check if adding 1 more monomer keeps the length below the threshold
        DOP += 1

    return DOP

def estimate_DOP_upper(monomer_smarts : ResidueSmarts, min_chain_len : int, min_DOP : int=3) -> int: # NOTE : as currently defined, this also subsumes the case when the estimate and calculated length are exactly equal
    '''Returns the smallest DOP for a set of monomers which yields a chain no shorter than the specified chain length'''
    return estimate_DOP_lower(monomer_smarts, min_chain_len, min_DOP=min_DOP) + 1 # by definition, this is just 1 monomer longer than the lower bound