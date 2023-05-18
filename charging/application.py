# Generic Imports
import numpy as np
from ast import literal_eval

# File I/O
from copy import copy
from pathlib import Path

# logging setup - will feed up to charging module parent logger
import logging
LOGGER = logging.getLogger(__name__)

# Typing and subclassing
from typing import Any, Optional
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

from . import TOOLKITS
from .types import ResidueChargeMap
from ..filetree import JSONifiable, JSONSerializable

from openmm.unit import elementary_charge
from openff.toolkit.topology.molecule import Molecule


# charge generation and application
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

def apply_averaged_res_chgs(mol : Molecule, residue_charges : ResidueChargeMap, in_place : bool=False) -> Optional[Molecule]:
    '''
    Takes an OpenFF Molecule and a residue-wise map of averaged partial charges and applies the mapped charges to the Molecule

    Can optionally specify whether to do charge assignment in-place (with "in_place" flag)
    -- If in_place, will apply to the Molecule passed and return None
    -- If not in_place, will create a copy, charge that, and return the resulting Molecule
    '''
    if in_place:
        _apply_averaged_res_chgs(mol, residue_charges)
    else:
        new_mol = copy(mol) # create replica of Molecule to leave charges undisturbed
        _apply_averaged_res_chgs(new_mol, residue_charges)
        return new_mol
    
# File I/O for format-specific decoding and deserialization
@dataclass
class ChargingParameters(JSONifiable):
    '''For recording the parameters used to assign a Polymer sets of partial charges'''
    overwrite_ff_xml   : bool
    overwrite_chg_mono : bool

    base_ff_path : Path 
    charge_methods : list[str]    # all charging methods which should be applied
    averaging_charge_method : str # method on which to base average charge calculations

    @staticmethod
    def serialize_json_dict(unser_jdict : dict[Any, Any]) -> dict[str, JSONSerializable]:
        '''For converting selfs __dict__ data into a form that can be serialized to JSON'''
        ser_jdict = {}
        for key, value in unser_jdict.items():
            if isinstance(value, Path):
                ser_jdict[key] = str(value)
            else:
                ser_jdict[key] = value

        return ser_jdict
    
    @staticmethod
    def unserialize_json_dict(ser_jdict : dict[str, JSONSerializable]) -> dict[Any, Any]:
        '''For de-serializing JSON-compatible data into a form that the __init__method can accept'''
        unser_jdict = {}
        for key, value in ser_jdict.items():
            if key == 'base_ff_path':
                unser_jdict[key] = Path(value)
            else:
                unser_jdict[key] = value

        return unser_jdict

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

def unserialize_monomer_json(ser_jdict : dict[str, JSONSerializable]) -> dict[Any, Any]:
    '''For unserializing charged residue maps in charged monomer JSON files'''
    unser_jdict = {}
    for key, value in ser_jdict.items():
        try: # TOSELF : need try-except instead of explicit check for "charges" key since objct hook is applied to ALL subdicts (not just main)
            unser_jdict[key] = { # convert string-keyed indices and charges back to numeric types
                int(substruct_id) : float(charge)
                    for substruct_id, charge in value.items()
            }
        except (ValueError, AttributeError):
            unser_jdict[key] = value
    
    return unser_jdict

# Molecule charging interface
class MolCharger(ABC):
    '''Base interface for defining various methods of generating and storing atomic partial charges'''
    @abstractproperty
    @classmethod
    def METHOD_NAME(cls):
        '''For setting the name of the method as a class attribute in child classes'''
        pass

    @abstractmethod
    def _charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Method for assigning molecular partial charges - concrete implementation in child classes'''
        pass

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Wraps charge method call with logging'''
        LOGGER.info(f'Assigning partial charges via the "{self.METHOD_NAME}" method')
        cmol = self._charge_molecule(uncharged_mol)
        LOGGER.info(f'Successfully assigned "{self.METHOD_NAME}" charges')

        return cmol


class ABE10Charger(MolCharger):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    METHOD_NAME = 'ABE10_exact'

    def _charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Concrete implementation for AM1-BCC-ELF10'''
        return generate_molecule_charges(uncharged_mol, toolkit='OpenEye Toolkit', partial_charge_method='am1bccelf10', force_match=True)

class EspalomaCharger(MolCharger):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    METHOD_NAME = 'Espaloma_AM1BCC'

    def _charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        return generate_molecule_charges(uncharged_mol, toolkit='Espaloma Charge Toolkit', partial_charge_method='espaloma-am1bcc', force_match=True)
    
class ABE10AverageCharger(MolCharger):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    METHOD_NAME = 'ABE10_averaged'

    def set_residue_charges(self, residue_charges : ResidueChargeMap):
        '''Slightly janky workaround to get initialization and the _charge_molecule interface to have the right number of args'''
        self.residue_charges = residue_charges

    def _charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        return apply_averaged_res_chgs(uncharged_mol, self.residue_charges, in_place=False)

CHARGER_REGISTRY = { # Keep a registry of all SDF charger implementations for convenience
    charger.METHOD_NAME : charger
        for charger in MolCharger.__subclasses__()
}