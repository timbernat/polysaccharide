# Typing and Subclassing
from typing import Any, Callable

# Cheminformatics
from openff.toolkit import Molecule, Topology
from rdkit import Chem

# Internal
from .rdmol.rdprops import copy_atom_metadata


def offmol_to_topo(offmol : Molecule, rdmol_func : Callable[[Chem.rdchem.Mol, Any], Chem.rdchem.Mol]=lambda x : x, allow_undefined_stereo : bool=True, hydrogens_are_explicit : bool=True) -> Topology:
    '''Accepts a Molecule and assembles it into a Topology. Optionally accepts a function which manipulates
    the RDKit representation of a Molecule (for instance, fragmenting it in a particular way)'''
    rdmol = rdmol_func(offmol.to_rdkit())
    frags = Chem.GetMolFrags(rdmol, asMols=True) # fragment Molecule in the case that multiple true molecules are present
    
    return Topology.from_molecules(
        Molecule.from_rdkit(frag, allow_undefined_stereo=allow_undefined_stereo, hydrogens_are_explicit=hydrogens_are_explicit)
            for frag in frags
    )

def to_rdkit_with_metadata(offmol : Molecule, preserve_type : bool=True) -> Chem.rdchem.Mol:
    '''Converts an OpenFF molecule to an RDKit molecule, preserving atomic metadata'''
    rdmol = offmol.to_rdkit()
    for i, offatom in enumerate(offmol.atoms):
        copy_atom_metadata(offatom, rdmol.GetAtomWithIdx(i), preserve_type=preserve_type)

    return rdmol
