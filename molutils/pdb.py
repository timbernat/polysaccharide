'''Tools for manipulating PDB structure and trajectory files'''
import re
from pathlib import Path
from functools import partial

from typing import Optional, Union
from ..filetree import filter_txt_by_condition


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