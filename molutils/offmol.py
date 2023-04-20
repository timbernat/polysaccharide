# Typing and Subclassing
from typing import Any, Callable

# Cheminformatics
from openff.toolkit import Molecule, Topology
from rdkit import Chem
from .rdmol.rdtypes import *


def offmol_to_topo(offmol : Molecule, rdmol_func : Callable[[RDMol, Any], RDMol]=lambda x : x,
                    allow_undefined_stereo : bool=True, hydrogens_are_explicit : bool=True) -> Topology:
    '''Accepts a Molecule and assembles it into a Topology. Optionally accepts a function which manipulates
    the RDKit representation of a Molecule (for instance, fragmenting it in a particular way)'''
    rdmol = rdmol_func(offmol.to_rdkit())
    frags = Chem.GetMolFrags(rdmol, asMols=True) # fragment Molecule in the case that multiple true molecules are present
    
    return Topology.from_molecules(
        Molecule.from_rdkit(frag, allow_undefined_stereo=allow_undefined_stereo, hydrogens_are_explicit=hydrogens_are_explicit)
            for frag in frags
    )
