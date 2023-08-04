'''For reading, writing, and querying atom map and isotope numbers in RDKit Atoms and Mols'''

from typing import Callable, Generator, Iterable, Optional, Union

from .rdtypes import RDMol, RDAtom
from ...general import optional_in_place


# READING FUNCTIONS
def get_isotopes(rdmol : RDMol, unique : bool=True) -> Union[set[int], list[int]]:
    '''Return all isotope IDs present in an RDMol. Can optioanlly return only the unique IDs'''
    isotope_ids = [atom.GetIsotope() for atom in rdmol.GetAtoms()]

    if unique:
        return set(isotope_ids)
    return isotope_ids

def get_ordered_map_nums(rdmol : RDMol) -> list[int]:
    '''Get assigned atom map numbers, in the same order as the internal RDMol atom IDs'''
    return [atom.GetAtomMapNum() for atom in rdmol.GetAtoms()]

def atom_ids_by_map_nums(rdmol : RDMol, *query_map_nums : list[int]) -> Generator[Optional[int], None, None]: # TODO : generalize this to handle case where multiple atoms have the same map num
    '''Returns the first occurences of the atom IDs of any number of atoms, indexed by atom map number'''
    present_map_nums : list[int] = get_ordered_map_nums(rdmol)
    for map_num in query_map_nums:
        try:
            yield present_map_nums.index(map_num)
        except ValueError: # if the provided map number is not found, return NoneType
            yield None


# WRITING FUNCTIONS
@optional_in_place    
def assign_ordered_atom_map_nums(rdmol : RDMol, start_from : int=1) -> None:
    '''Assigns atom's index as its atom map number for all atoms in an RDmol
    Can optionally specify what value to begin counting from (by default 1)'''
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + start_from) # NOTE that starting from anything below 1 will cause an atom somewhere to be mapped to 0 (i.e. not mapped)

@optional_in_place    
def assign_atom_map_nums_from_ref(rdmol : RDMol, ref : Iterable[int]) -> None:
    '''Assigns atom map numbers using an external collection of values'''
    for atom, map_num in zip(rdmol.GetAtoms(), ref): # TODO : add some way to check that lengths match (may be generator-like)
        atom.SetAtomMapNum(map_num) 

def assign_contiguous_atom_map_nums(*rdmols : Iterable[RDMol], start_from : int=1, in_place : bool=False) -> Optional[list[RDMol]]: 
    '''Assign sequential numbering to a collection of molecules such that their map numbers span a contiguous range of integers
    Can optionally specify what value to begin counting from (by default 1)'''
    new_mols = []

    map_num_counter = start_from # initialize index tracker with starting index
    for rdmol in rdmols:
        new_mol = assign_ordered_atom_map_nums(rdmol, start_from=map_num_counter, in_place=in_place) # note : will be assigned NoneType if in-place
        if not in_place:
            new_mols.append(new_mol)
        map_num_counter += rdmol.GetNumAtoms()

    if new_mols:
        return new_mols

# CLEARING FUNCTIONS
@optional_in_place
def clear_atom_map_nums(rdmol : RDMol) -> None:
    '''Removes atom map numbers from all atoms in an RDMol'''
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(0)

@optional_in_place
def clear_atom_isotopes(rdmol : RDMol) -> None:
    '''Removes isotope numbers from all atoms in an RDMol'''
    for atom in rdmol.GetAtoms():
        atom.SetIsotope(0)


# NEIGHBOR ATOM INFO
def _neighbor_factory_by_condition(condition : Callable[[RDAtom], bool]) -> Callable[[RDAtom], Generator[RDAtom, None, None]]:
    '''Factory function for generating neighbor-search functions over RDAtoms by a boolean condition'''
    def neighbors_by_condition(atom : RDAtom) -> Generator[RDAtom, None, None]:
        '''Generate all neighboring atoms satisfying a condition'''
        for nb_atom in atom.GetNeighbors():
            if condition(nb_atom):
                yield nb_atom

    return neighbors_by_condition

def _has_neighbor_factory_by_condition(condition : Callable[[RDAtom], bool]) -> Callable[[RDAtom], bool]:
    '''Factory function for generating neighbor-search functions over RDAtoms by a boolean condition'''
    def has_neighbors_by_condition(atom : RDAtom) -> bool:
        '''Identify if any neighbors of an atom satisfy some condition'''
        return any(
            condition(nb_atom)
                for nb_atom in atom.GetNeighbors()
        )

    return has_neighbors_by_condition