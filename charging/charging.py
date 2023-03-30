import numpy as np
from typing import Optional
from copy import copy

from openmm.unit import elementary_charge
from openff.toolkit.topology.molecule import Molecule

from . import TOOLKITS
from .types import ResidueChargeMap


# FUNCTIONS
def generate_molecule_charges(mol : Molecule, toolkit : str='OpenEye Toolkit', partial_charge_method : str='am1bccelf10', force_match : bool=True) -> Molecule:
    '''Takes a Molecule object and computes partial charges with AM1BCC using toolkit method of choice. Returns charged molecule'''
    tk_reg = TOOLKITS.get(toolkit)
    mol.assign_partial_charges(partial_charge_method=partial_charge_method, toolkit_registry=tk_reg)
    charged_mol = mol # rename for clarity

    # charged_mol.generate_conformers( # get some conformers to run elf10 charge method. By default, `mol.assign_partial_charges`...
    #     n_conformers=10,             # ...uses 500 conformers, but we can generate and use 10 here for demonstration
    #     rms_cutoff=0.25 * unit.angstrom,
    #     make_carboxylic_acids_cis=True,
    #     toolkit_registry=tk_reg
    # ) # very slow for large polymers! 

    if force_match:
        for atom in charged_mol.atoms:
            assert(atom.metadata['already_matched'] == True)
        
    return charged_mol 

def _apply_averaged_res_chgs(mol : Molecule, residue_charges : ResidueChargeMap) -> None:
    '''Takes an OpenFF Molecule and a residue-wise map of averaged partial charges and applies the mapped charges to the Molecule'''
    new_charges = [
        residue_charges[atom.metadata['residue_name']][atom.metadata['substructure_id']]
            for atom in mol.atoms
    ]
    new_charges = np.array(new_charges) * elementary_charge # convert to unit-ful array (otherwise assignment won't work)
    mol.partial_charges = new_charges

def apply_averaged_res_chgs(mol : Molecule, residue_charges : ResidueChargeMap, inplace : bool=False) -> Optional[Molecule]:
    '''
    Takes an OpenFF Molecule and a residue-wise map of averaged partial charges and applies the mapped charges to the Molecule

    Can optionally specify whether to do charge assignment in-place (with "inplace" flag)
    -- If inplace, will apply to the Molecule passed and return None
    -- If not inplace, will create a copy, charge that, and return the resulting Molecule
    '''
    if inplace:
        _apply_averaged_res_chgs(mol, residue_charges)
    else:
        new_mol = copy(mol) # create replica of Molecule to leave charges undisturbed
        _apply_averaged_res_chgs(new_mol, residue_charges)
        return new_mol