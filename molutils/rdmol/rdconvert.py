'''For conversion of RDMols back and forth between different format encodings - often imbues a desired side effect (such as 2D-projection)'''

# Typing
from typing import Union
from .rdtypes import RDMol

# Subclassing
import copy
from abc import ABC, abstractmethod, abstractproperty

# Cheminformatics
from rdkit import Chem
from . import rdprops


class RDConverter(ABC):
    '''For converting an existing RDKit Molecule to and from a particular format to gain new properties'''
    @abstractproperty
    @classmethod
    def TAG(cls):
        pass

    @abstractmethod
    def convert(self, rdmol : RDMol) -> RDMol:
        pass

class SMARTSConverter(RDConverter):
    TAG = 'SMARTS'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmarts(Chem.MolToSmarts(rdmol))

class SMILESConverter(RDConverter):
    TAG = 'SMILES'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmiles(Chem.MolToSmiles(rdmol), sanitize=False)
    
class CXSMARTSConverter(RDConverter):
    '''Similar to SMARTSConverter but preserves the 3D structure'''
    TAG = 'CXSMARTS'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmarts(Chem.MolToCXSmarts(rdmol))

class CXSMILESConverter(RDConverter):
    '''Similar to SMILESConverter but preserves the 3D structure'''
    TAG = 'CXSMILES'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmiles(Chem.MolToCXSmiles(rdmol), sanitize=False)

class InChIConverter(RDConverter): # TOSELF : this does not preserve atom map num ordering (how to incorporate AuxInfo?)
    TAG = 'InChI'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.AddHs(Chem.MolFromInchi(Chem.MolToInchi(rdmol), removeHs=False, sanitize=False))
    
class JSONConverter(RDConverter):
    TAG = 'JSON'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.rdMolInterchange.JSONToMols(Chem.MolToJSON(rdmol))[0]


# define registry for convenient lookup once all child classes are defined above
RDCONVERTER_REGISTRY = { # keep easily accessible record of all available converter types
    child.TAG : child()
        for child in RDConverter.__subclasses__()
}

# functons for applying various converters to RDMols
def flattened_rdmol_legacy(rdmol : RDMol, converter : Union[str, RDConverter]='SMARTS') -> RDMol:
    '''Returns a flattened version of an RDKit molecule for 2D representation'''
    if isinstance(converter, str): # simplifies external function calls (don't need to be aware of underlying RDConverter class explicitly)
        converter = RDCONVERTER_REGISTRY[converter] # perform lookup if only name is passed

    orig_rdmol = copy.deepcopy(rdmol) # create copy to avoid mutating original
    rdprops.assign_ordered_atom_map_nums(orig_rdmol, in_place=True) # need atom map numbers to preserve positional mapping in SMARTS

    flat_mol = converter.convert(orig_rdmol) # apply convert for format interchange
    if set(atom.GetAtomMapNum() for atom in flat_mol.GetAtoms()) == {0}: # hacky workaround for InChI and other formats which discard atom map number - TODO : fix this terriblenesss
        rdprops.assign_ordered_atom_map_nums(flat_mol, in_place=True)

    rdprops.copy_rd_props(to_rdobj=flat_mol, from_rdobj=orig_rdmol) # clone molecular properties
    for new_atom in flat_mol.GetAtoms():
        rdprops.copy_rd_props(to_rdobj=new_atom, from_rdobj=orig_rdmol.GetAtomWithIdx(new_atom.GetAtomMapNum() - 1)) # -1 is to account for 0-index exclusion in map numbering; atom numbered "0" is invalid

    del orig_rdmol # mark copy for garbage collection now that it has served its purpose
    return flat_mol

def flattened_rdmol(rdmol : RDMol, converter : Union[str, RDConverter]='SMARTS') -> RDMol:
    '''Returns a flattened version of an RDKit molecule for 2D representation'''
    if isinstance(converter, str): # simplifies external function calls (don't need to be aware of underlying RDConverter class explicitly)
        converter = RDCONVERTER_REGISTRY[converter] # perform lookup if only name is passed

    flat_mol = converter.convert(rdmol) # apply convert for format interchange
    atom_mapping = rdmol.GetSubstructMatch(flat_mol) # map 
    if (not atom_mapping) or (len(atom_mapping) != rdmol.GetNumAtoms()):
        raise ValueError('Substructure match failed') # TODO : make this a SubstructureMatchFailedError, from polymer.exceptions
    
    for orig_idx, new_atom in zip(atom_mapping, flat_mol.GetAtoms()):
        orig_atom = rdmol.GetAtomWithIdx(orig_idx)
        rdprops.copy_rd_props(to_rdobj=new_atom, from_rdobj=orig_atom)

    return flat_mol
