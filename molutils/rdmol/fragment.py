from typing import Any, Callable

# Functional methods
from functools import partial
from operator import xor

# Cheminformatics
from openff.toolkit import Molecule
from rdkit import Chem
from .rdtypes import *


# methods for obtaining fragment bond indices
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

# fragmentation methods
def fragment_by_bond_indices(rdmol : RDMol, bond_ids : list[int], **frag_args) -> tuple[RDMol, ...]:
    '''Accepts an RDKit Mol and indices of bonds and returns the mol fragments produced by breaking those bonds'''
    rdmol_cut = Chem.FragmentOnBonds(rdmol, bond_ids)
    return Chem.GetMolFrags(rdmol_cut, asMols=True, **frag_args)