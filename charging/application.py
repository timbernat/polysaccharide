import numpy as np
from copy import copy
from ast import literal_eval

from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod, abstractproperty

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
    
def load_matched_charged_molecule(sdf_path : Path, assume_ordered : bool=True) -> Molecule:
    '''Special load instruction for charged SDF files - necessary to smoothly remap atom metadata from properties'''
    cmol_matched = Molecule.from_file(sdf_path)
    metadata = literal_eval(cmol_matched.properties['metadata']) # needed to de-stringify list

    if assume_ordered:  # assumes metadata list is in order by atom ID
        for (atom, mdat) in zip(cmol_matched.atoms, metadata):
            for (key, value) in mdat.items():
                atom.metadata[key] = value 
    else: # slightly clunkier but doesn't assume anything about ordering (more robust)
        metadata_map = {mdat['pdb_atom_id'] : mdat for mdat in metadata} 
        for atom in cmol_matched.atoms:
            mdat = metadata_map[atom.molecule_atom_index]
            for (key, value) in mdat.items():
                atom.metadata[key] = value 
    
    cmol_matched.properties.pop('metadata') # remove metadata packed for notational cleanliness
    return cmol_matched

# Molecule charging interface
class MolCharger(ABC):
    '''Base interface for defining various methods of generating and storing atomic partial charges'''
    @classmethod
    @abstractproperty
    def TAG(cls):
        '''For setting the name of the method as a class attribute in child classes'''
        pass

    @abstractmethod
    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Concrete implementation for producing molecule partial charges'''
        pass

class ABE10Charger(MolCharger):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    TAG = 'ABE10_exact'

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Concrete implementation for AM1-BCC-ELF10'''
        return generate_molecule_charges(uncharged_mol, toolkit='OpenEye Toolkit', partial_charge_method='am1bccelf10', force_match=True)

class EspalomaCharger(MolCharger):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    TAG = 'Espaloma_AM1BCC'

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        return generate_molecule_charges(uncharged_mol, toolkit='Espaloma Charge Toolkit', partial_charge_method='espaloma-am1bcc', force_match=True)
    
class ABE10AverageCharger(MolCharger):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    TAG = 'ABE10_averaged'

    def set_residue_charges(self, residue_charges : ResidueChargeMap):
        '''Slightly janky workaround to get initialization and the charge_molecule interface to have the right number of args'''
        self.residue_charges = residue_charges

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        return apply_averaged_res_chgs(uncharged_mol, self.residue_charges, inplace=False)

CHARGER_REGISTRY = { # Keep a registry of all SDF charger implementations for convenience
    charger.TAG : charger
        for charger in MolCharger.__subclasses__()
}