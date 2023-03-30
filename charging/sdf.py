from pathlib import Path
from ast import literal_eval
from abc import ABC, abstractmethod, abstractproperty

from openff.toolkit.topology.molecule import Molecule

from .types import ResidueChargeMap
from .charging import generate_molecule_charges, apply_averaged_res_chgs


# FUNCTIONS
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

    return cmol_matched


# CLASSES
# Molecule-charging-to-SDF interface
class SDFChargerBase(ABC):
    '''Interface for defining various methods of generating and storing atomic partial charges'''
    @classmethod
    @abstractproperty
    def TAG(self):
        '''For setting the name of the method as a class attribute in child classes'''
        pass

    @abstractmethod
    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Concrete implementation for producing molecule partial charges'''
        pass

    def generate_sdf(self, uncharged_mol : Molecule, sdf_path : Path) -> Molecule:
        '''Accepts an uncharged Molecule and a Path to an SDF, generates an SDF file of the corresponding charged molecule, returns the charged molecule'''
        assert(sdf_path.suffix == '.sdf')

        cmol = self.charge_molecule(uncharged_mol)
        cmol.properties['metadata'] = [atom.metadata for atom in cmol.atoms] # need to store metadata as separate property, since SDF does not preserved metadata atomwise
        cmol.to_file(str(sdf_path), file_format='SDF')

        return cmol

class ABE10Charger(SDFChargerBase):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    TAG = 'ABE10_exact'

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        '''Concrete implementation for AM1-BCC-ELF10'''
        return generate_molecule_charges(uncharged_mol, toolkit='OpenEye Toolkit', partial_charge_method='am1bccelf10', force_match=True)

class EspalomaCharger(SDFChargerBase):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    TAG = 'Espaloma_AM1BCC'

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        return generate_molecule_charges(uncharged_mol, toolkit='Espaloma Charge Toolkit', partial_charge_method='espaloma-am1bcc', force_match=True)
    
class ABE10AverageCharger(SDFChargerBase):
    '''Charger class for AM1-BCC-ELF10 exact charging'''
    TAG = 'ABE10_averaged'

    def set_residue_charges(self, residue_charges : ResidueChargeMap):
        '''Slightly janky workaround to get initialization and the charge_molecule interface to have the right number of args'''
        self.residue_charges = residue_charges

    def charge_molecule(self, uncharged_mol : Molecule) -> Molecule:
        return apply_averaged_res_chgs(uncharged_mol, self.residue_charges, inplace=False)


# Keep a registry of all SDF charger implementations for convenience
CHARGER_REGISTRY = {
    charger.TAG : charger
        for charger in SDFChargerBase.__subclasses__()
}