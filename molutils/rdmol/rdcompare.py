# Typing and Subclassing
import copy # revise diff_mol generation to eschew this import
from typing import Union

# Cheminformatics
from rdkit import Chem

# Internal
from .rdtypes import *
from .rdconvert import RDConverter, RDCONVERTER_REGISTRY


def copy_rd_props(from_rdobj : Union[RDAtom, RDMol], to_rdobj : Union[RDAtom, RDMol]) -> None:
    '''For copying properties between a pair of RDKit Atoms or Mols'''
    # NOTE : will avoid use of GetPropsAsDict() to avoid errors from ridiculously restrictive C++ typing
    # for prop, prop_val in from_rdobj.GetPropsAsDict().items():
    #     to_rdobj.SetProp(prop, prop_val)

    for prop in from_rdobj.GetPropNames():
        to_rdobj.SetProp(prop, from_rdobj.GetProp(prop))

def assign_ordered_atom_map_nums(rdmol : RDMol) -> None:
    '''Assigns atom's id to its atom map number for all atoms in an RDmol'''
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()) # need atom map numbers to preserve positional mapping in SMARTS
        # atom.SetAtomMapNum(atom.GetIdx() + 1) # need atom map numbers to preserve positional mapping in SMARTS; "+1" avoids mapping any atoms to 0

def flattened_rdmol(rdmol : RDMol, converter : Union[str, RDConverter]='SMARTS') -> RDMol:
    '''Returns a flattened version of an RDKit molecule for 2D representation'''
    if isinstance(converter, str): # simplifies external function calls (don't need to be aware of underlying RDConverter class explicitly)
        converter = RDCONVERTER_REGISTRY[converter] # perform lookup if only name is passed

    orig_rdmol = copy.deepcopy(rdmol) # create copy to avoid mutating original
    assign_ordered_atom_map_nums(orig_rdmol) # need atom map numbers to preserve positional mapping in SMARTS

    flat_mol = converter.convert(orig_rdmol) # apply convert for format interchange
    if set(atom.GetAtomMapNum() for atom in flat_mol.GetAtoms()) == {0}: # hacky workaround for InChI and other formats which discard atom map number - TODO : fix this terriblenesss
        assign_ordered_atom_map_nums(flat_mol)

    copy_rd_props(to_rdobj=flat_mol, from_rdobj=orig_rdmol) # clone molecular properties
    for new_atom in flat_mol.GetAtoms():
        copy_rd_props(to_rdobj=new_atom, from_rdobj=orig_rdmol.GetAtomWithIdx(new_atom.GetAtomMapNum()))

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
