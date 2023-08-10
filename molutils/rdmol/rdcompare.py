'''Utilities for comparing partal charges between pairs of RDMols'''

# Generic imports
import copy # revise diff_mol generation to eschew this import
import matplotlib.pyplot as plt

from ...general import GREEK_UPPER
from .rdkdraw import rdmol_prop_heatmap_colorscaled

# Typing and Subclassing
from typing import Union
from matplotlib.colors import Colormap

from .rdtypes import RDMol

# Cheminformatics
from rdkit import Chem
from . import rdconvert


def difference_rdmol_legacy(rdmol_1 : RDMol, rdmol_2 : RDMol, prop : str='PartialCharge', remove_map_nums : bool=True) -> RDMol:
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

def difference_rdmol(rdmol_1 : RDMol, rdmol_2 : RDMol, prop : str='PartialCharge', remove_map_nums : bool=True) -> RDMol:
    '''
    Takes two RDKit Mols (presumed to have the same structure and atom map numbers) and the name of a property 
    whose partial charges are the differences betwwen the two Mols' charges (atomwise)
    
    Assumes that the property in question is numeric (i.e. can be interpreted as a float)
    '''
    diff_mol = copy.deepcopy(rdmol_1) # duplicate first molecule as template
    atom_mapping = diff_mol.GetSubstructMatch(rdmol_2) # map 
    if (not atom_mapping) or (len(atom_mapping) != diff_mol.GetNumAtoms()):
        raise ValueError('Substructure match failed') # TODO : make this a SubstructureMatchFailedError, from polymer.exceptions

    all_deltas = []
    for rdatom_idx_1, rdatom_2 in zip(atom_mapping, rdmol_2.GetAtoms()):
        rdatom_1 = rdmol_1.GetAtomWithIdx(rdatom_idx_1)
        diff_atom = diff_mol.GetAtomWithIdx(rdatom_idx_1) # same index, since it is a deep copy

        delta = rdatom_1.GetDoubleProp(prop) - rdatom_2.GetDoubleProp(prop)
        diff_atom.SetDoubleProp(f'Delta{prop}', delta)
        all_deltas.append(delta)

        diff_atom.ClearProp(prop) # reset property value from original copy to avoid confusion
        if remove_map_nums:
            diff_atom.ClearProp('molAtomMapNumber') # Remove atom map num for greater visual clarity when drawing

    diff_mol.SetProp(f'Delta{prop}s', str(all_deltas)) # label stringified version of property list (can be de-stringified via ast.literal_eval)
    diff_mol.SetDoubleProp(f'Delta{prop}Min', min(all_deltas)) # label minimal property value for ease of reference
    diff_mol.SetDoubleProp(f'Delta{prop}Max', max(all_deltas)) # label maximal property value for ease of reference

    return diff_mol
