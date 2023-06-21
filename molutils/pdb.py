'''Tools for manipulating PDB structure and trajectory files'''

# Generic imports
import re
from pathlib import Path
from functools import partial

# Typing and subclassing
from typing import Optional, Union

# Cheminformatics
from rdkit import Chem

# Custom imports
from ..filetree import filter_txt_by_condition
from ..general import asstrpath


# fastest by benchmark on 12,188 atom, 2,500 frame solvated PNIPAAm
def has_water(line : str) -> bool: # 8.247 sec in benchmark
    '''Check if a text line in a file contains a water residue'''
    return ('wat' in line) or ('HOH' in line)

# def has_water(line : str) -> bool: # 24.281 sec in benchmark 
#     '''Check if a text line in a file contains a water residue'''
#     return any(word in line for word in ('wat', 'HOH'))

# def has_water(line : str) -> bool: # 77.254 sec in benchmark
#     '''Check if a text line in a file contains a water residue'''
#     regex = re.compile('(wat)|(HOH)')
#     return bool(re.search(regex, line))

# strip_water = partial(filter_txt_by_regex, condition=has_water, postfix='dewatered', inclusive=False)
def strip_water(pdb_in : Path, pdb_out : Optional[Path]=None) -> Path: # TODO : generalize to arbitrary solvent using Solvent properties?
    '''Create a copy of a trajectory PDB with all water residues removed
    Returns path to copied PDB (in same dir as original if pdb_out not explicitly specified)'''
    return filter_txt_by_condition(pdb_in, out_txt_path=pdb_out, condition=has_water, inclusive=False, postfix='dewatered', return_filtered_path=True)

def pdb_water_atoms_to_hetatoms(pdb_path : Union[str, Path], output_path : Optional[Union[str, Path]]=None) -> None:
    '''Ensures that all water atoms in a PDB are correctly labelled as heteratoms'''
    pdb_path_str = asstrpath(pdb_path) # force path to be string for RDKit-compatibility
    rdmol = Chem.MolFromPDBFile(pdb_path_str, removeHs=False)
    
    for atom in rdmol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo() 
        if pdb_info.GetResidueName() == 'HOH':
            pdb_info.SetIsHeteroAtom(True)

    if output_path is None:
        output_path = pdb_path
    output_path_str = asstrpath(output_path)

    Chem.MolToPDBFile(rdmol, output_path_str)