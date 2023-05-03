'''Tools for manipulating PDB structure and trajectory files'''
import re
from functools import partial
from ..filetree import filter_txt_by_regex


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

strip_water = partial(filter_txt_by_regex, condition=has_water, postfix='dewatered', inclusive=False)