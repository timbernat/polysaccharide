# Typing and Subclassing
import copy # revise diff_mol generation to eschew this import
from typing import Any, Callable, Union
from abc import ABC, abstractmethod, abstractproperty

# Functional methods
from functools import partial
from polymer_utils.general import xor

# Cheminformatics
from openff.toolkit import Molecule
from rdkit import Chem

RDMol = Chem.rdchem.Mol
RDAtom = Chem.rdchem.Atom
RDBond = Chem.rdchem.Bond

# Converting to and from various molecule encodings - enables transformation of 3D conformers into 2D structures for representation
class RDConverter(ABC):
    '''For converting an existing RDKit Molecule to and from a particular format to gain new properties'''
    @classmethod
    @abstractproperty
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
    
RDCONVERTER_REGISTRY = { # keep easily accessible record of all available converter types
    child.TAG : child()
        for child in RDConverter.__subclasses__()
}

# Flattening and comparison of RDkit Molecules
def _copy_rd_props(from_rdobj : Union[RDAtom, RDMol], to_rdobj : Union[RDAtom, RDMol]) -> None:
    '''For copying properties between a pair of RDKit Atoms or Mols'''
    # NOTE : will avoid use of GetPropsAsDict() to avoid errors from ridiculously restrictive C++ typing
    # for prop, prop_val in from_rdobj.GetPropsAsDict().items():
    #     to_rdobj.SetProp(prop, prop_val)

    for prop in from_rdobj.GetPropNames():
        to_rdobj.SetProp(prop, from_rdobj.GetProp(prop))

def _assign_ordered_atom_map_nums(rdmol : RDMol) -> None:
    '''Assigns atom's id to its atom map number for all atoms in an RDmol'''
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()) # need atom map numbers to preserve positional mapping in SMARTS

def flattened_rmdol(rdmol : RDMol, converter : Union[str, RDConverter]='SMARTS') -> RDMol:
    '''Returns a flattened version of an RDKit molecule for 2D representation'''
    if isinstance(converter, str): # simplifies external function calls (don't need to be aware of underlying RDConverter class explicitly)
        converter = RDCONVERTER_REGISTRY[converter] # perform lookup if only name is passed

    orig_rdmol = copy.deepcopy(rdmol) # create copy to avoid mutating original
    _assign_ordered_atom_map_nums(orig_rdmol) # need atom map numbers to preserve positional mapping in SMARTS

    flat_mol = converter.convert(orig_rdmol) # apply convert for format interchange
    if set(atom.GetAtomMapNum() for atom in flat_mol.GetAtoms()) == {0}: # hacky workaround for InChI and other formats which discard atom map number - TODO : fix this terriblenesss
        _assign_ordered_atom_map_nums(flat_mol)

    _copy_rd_props(to_rdobj=flat_mol, from_rdobj=orig_rdmol) # clone molecular properties
    for new_atom in flat_mol.GetAtoms():
        _copy_rd_props(to_rdobj=new_atom, from_rdobj=orig_rdmol.GetAtomWithIdx(new_atom.GetAtomMapNum()))

    del orig_rdmol # free up memory used by copy
    return flat_mol

def difference_rdmol(rdmol_1 : Chem.rdchem.Mol, rdmol_2 : Chem.rdchem.Mol, prop : str='PartialCharge', remove_map_nums : bool=True) -> Chem.rdchem.Mol:
    '''
    Takes two RDKit Mols (presumed to have the same structure and atom map numbers) and the name of a property 
    whose partial charges are the differences betwwen the two Mols' charges (atomwise)
    
    Assumes that the property in question is numeric (i.e. can be interpreted as a float)
    '''
    diff_mol = copy.deepcopy(rdmol_1) # duplicate first molecule as template
    all_deltas = []
    for atom in diff_mol.GetAtoms():
        rdatom_1 = rdmol_1.GetAtomWithIdx(atom.GetAtomMapNum())
        rdatom_2 = rdmol_2.GetAtomWithIdx(atom.GetAtomMapNum())
        delta = rdatom_1.GetDoubleProp(prop) - rdatom_2.GetDoubleProp(prop)

        atom.SetDoubleProp(f'Delta{prop}', delta)
        all_deltas.append(delta)

        atom.ClearProp(prop) # reset property value from original copy to avoid confusion
        if remove_map_nums:
            atom.ClearProp('molAtomMapNumber') # Remove atom map num for greater visual clarity when drawing

    diff_mol.SetProp(f'Delta{prop}s', str(all_deltas)) # label stringified version of property list (can be de-stringified via ast.literal_eval)
    diff_mol.SetDoubleProp(f'Delta{prop}Min', min(all_deltas)) # label minimal property value for ease of reference
    diff_mol.SetDoubleProp(f'Delta{prop}Max', max(all_deltas)) # label maximal property value for ease of reference

    return diff_mol

# Fragmentation
def inter_monomer_bond_indices(offmol : Molecule) -> list[int]:
    '''Return the bonds which bridge two distinct monomer fragments (as labelled by graph match)'''
    return [
        offmol.to_rdkit().GetBondBetweenAtoms(bond.atom1_index, bond.atom2_index).GetIdx()
            for bond in offmol.bonds
                if bond.atom1.metadata['residue_number'] != bond.atom2.metadata['residue_number']
    ]

def oligomers_by_length(offmol : Molecule, frag_len : int=100) -> list[int]:
    '''Return the bonds which are between consecutive groups of monomers which have fewer than frag_len atoms combined'''
    inter_mono_bonds = inter_monomer_bond_indices(offmol)
    num_cuts = len(offmol.bonds) // frag_len

    return sorted((i for i in inter_mono_bonds), key=lambda x : x % frag_len)[:num_cuts]

def bond_ids_by_cond(rdmol : RDMol, bond_cond : Callable[[RDBond, Any], bool]) -> tuple[int]:
    '''Return IDs of all bonds which satisfy some binary condition'''
    return (bond.GetIdx() for bond in rdmol.GetBonds() if bond_cond(bond))

between_numbered_atoms = partial(bond_ids_by_cond, bond_cond=lambda bond : xor(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))


def fragment_by_bond_indices(rdmol : RDMol, bond_ids : list[int]) -> tuple[RDMol, ...]:
    '''Accepts an RDKit Mol and indices of bonds and returns the mol fragments produced by breaking those bonds'''
    rdmol_cut = Chem.FragmentOnBonds(rdmol, bond_ids)
    return Chem.GetMolFrags(rdmol_cut, asMols=True)
