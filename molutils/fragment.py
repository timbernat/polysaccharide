'''Functions for slicing and dicing RDkit molecules'''

# Typing and subclassing
from typing import Any, Callable, Iterable
from .rdmol.rdtypes import RDMol, RDBond

# Functional methods
from functools import partial
from operator import xor

# Cheminformatics
from openff.toolkit import Molecule
from rdkit import Chem
from .rdmol.rdbond import hydrogenate_rdmol_ports


# OpenFF-specific methods for obtaining fragment bond indices
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

# RDKit-specific methods for obtaining fragment bond indices
def bond_ids_by_cond(rdmol : RDMol, bond_cond : Callable[[RDBond, Any], bool]) -> tuple[int]:
    '''Return IDs of all bonds which satisfy some binary condition'''
    return (bond.GetIdx() for bond in rdmol.GetBonds() if bond_cond(bond))

between_numbered_atoms = partial(bond_ids_by_cond, bond_cond=lambda bond : xor(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))

# residue-labelling methods
def residues_in_fragment_offmol(offmol : Molecule, frag_idxs : Iterable[int]) -> set[str]:
    '''Returns all residues that the atoms specified by a collection of atom ids are members of'''
    return set(
        offmol.atoms[idx].metadata['residue_name']
            for idx in frag_idxs
                if idx < offmol.n_atoms # explicitly ensure atom is within range (relevant when fragmenting on bonds in RDKit, where extra wild atoms are added)
    )

def residues_in_fragment_rdmol(rdmol : RDMol, frag_idxs : Iterable[int]) -> set[str]:
    '''Returns all residues that the atoms specified by a collection of atom ids are members of'''
    return set(
        rdmol.GetAtomWithIdx(idx).GetProp('residue_name')
            for idx in frag_idxs
                if idx < rdmol.GetNumAtoms() # explicitly ensure atom is within range (relevant when fragmenting on bonds in RDKit, where extra wild atoms are added)
    )
# fragmentation methods
def fragment_by_bond_indices(rdmol : RDMol, bond_ids : list[int], add_ports : bool=True, **frag_args) -> tuple[RDMol, ...]:
    '''Accepts an RDKit Mol and indices of bonds and returns the mol fragments produced by breaking those bonds'''
    rdmol_cut = Chem.FragmentOnBonds(rdmol, bond_ids, addDummies=add_ports) # by default, will add port atoms to cut locations to preserve valence
    return Chem.GetMolFrags(rdmol_cut, **frag_args)

def monomer_frags_with_residues(offmol : Molecule, hydrogenate_ports : bool=False) -> dict[RDMol, str]:
    '''Takes an RDKit molecule (with atomwise metadata) and returns a dict keyed by molecule fragments 
    with values corresponding to the identified residue each fragment is a member of'''
    frag_idx_list = [] # needed to capture the atom indices belonging to each fragment
    bond_cut_ids = inter_monomer_bond_indices(offmol)
    frag_mols = fragment_by_bond_indices(offmol.to_rdkit(), bond_cut_ids, asMols=True, fragsMolAtomMapping=frag_idx_list)

    frags_with_residues = {}
    for (frag_mol, frag_idxs) in zip(frag_mols, frag_idx_list):
        residues_in_frag = residues_in_fragment_offmol(offmol, frag_idxs)
        if len(residues_in_frag) != 1:
            raise ValueError(f'Fragment at indices {frag_idxs} does not encompass a single residue')
        
        if hydrogenate_ports: # add hydrogens to ports if desired
            hydrogenate_rdmol_ports(frag_mol)
        frags_with_residues[frag_mol] = residues_in_frag.pop()
    
    return frags_with_residues