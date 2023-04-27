# Custom Imports
from . import general, filetree, simulation
from .logutils import timestamp_now
from .solvation.packmol import packmol_solvate_wrapper
from .charging.averaging import write_lib_chgs_from_mono_data
from .charging.application import load_matched_charged_molecule, unserialize_monomer_json
from .graphics.rdkdraw import compare_chgd_rdmols

# Typing and Subclassing
from .charging.application import MolCharger
from .solvation.solvent import Solvent
from .extratypes import SubstructSummary, Figure, Axes, ndarray
from .charging.types import ResidueChargeMap

from typing import Any, Callable, ClassVar, Iterable, Optional, Union

# File I/O
import copy
from pathlib import Path
from shutil import copyfile, copytree
import json, pickle

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# Cheminformatics and MD
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol

# Molecular Dynamics
from openff.toolkit import ForceField
from openff.interchange import Interchange
from openff.toolkit.topology import Topology, Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import LibraryChargeHandler

from openmm.openmm import MonteCarloBarostat
from openmm import LangevinMiddleIntegrator
from openmm.unit import angstrom, nanometer

# Custom Exceptions
class MissingStructureData(Exception):
    pass

class MissingMonomerData(Exception):
    pass

class MissingMonomerUnchargedData(Exception):
    pass

class MissingChargedMonomerData(Exception):
    pass

class MissingForceFieldData(Exception):
    pass

class AlreadySolvatedError(Exception):
    pass


# Polymer representation class
def create_subdir_properties(cls):
    '''For dynamically adding all subdirectories as accessible property fields - cleans up namespace for Polymer'''
    for dir_name in cls._SUBDIRS:
        def make_prop(subdir_name : str):
            return property(fget=lambda self : self.subdirs[subdir_name])
        
        setattr(cls, dir_name, make_prop(dir_name))
    return cls

@create_subdir_properties
class Polymer:
    '''For representing standard directory structure and requisite information for polymer structures, force fields, and simulations'''
    _SUBDIRS : ClassVar[tuple[str, ...]] = ( # directories with these names will be present in all polymer directories by standard
        'structures',
        'monomers',
        'SDF',
        'FF',
        'MD',
        'checkpoint',
        'logs'
    )

# CONSTRUCTION 
    def __init__(self, parent_dir : Path, mol_name : str, exclusion : float=1*nanometer) -> None:
        '''Initialize core directory, file paths, and attributes'''
        self.parent_dir : Path = parent_dir
        self.mol_name   : str = mol_name
        self.exclusion  : float = exclusion

        # Derived attributes
        self.base_mol_name : str = self.mol_name # will be updated if ever solvated (not useful otherwise)
        self.path : Path = parent_dir / mol_name

        # "Public" attributes which are governed by methods (should not be set at init)
        self.solvent : Solvent = None # kept default as NoneType rather than empty string for more intuitive solvent search with unsolvated molecules
        self.charges : dict[str, ndarray[float]] = {}
        self.charge_method : str = None
        
        self.ff_file              : Optional[Path]   = None # .OFFXML
        self.monomer_file         : Optional[Path]   = None # .JSON
        self.monomer_file_chgd    : Optional[Path]   = None # .JSON
        self.structure_file       : Optional[Path]   = None # .PDB
        self.structure_files_chgd : dict[str, Path]  = {} # dict of .SDF
        self.simulation_paths     : dict[Path, Path] = {}

        # "Private" attributes
        self._off_topology : Topology = None
        self._offmol       : Molecule = None
        
        self._off_topology_unmatched : Topology = None
        self._offmol_unmatched       : Molecule = None

        # Filetree creation
        self.subdirs         : dict[str, Path] = {} # reference specific directories corresponding to the standard class-wide directories
        self.subdirs_reverse : dict[str, Path] = {} # also create reverse lookup for getting attributes for parent directory paths
        for dirname in Polymer._SUBDIRS:
            subdir = self.path/dirname
            self.subdirs[dirname] = subdir
            self.subdirs_reverse[subdir] = dirname

        self.checkpoint_path : Path = self.checkpoint/f'{self.mol_name}_checkpoint.pkl' # NOTE : must be called AFTER subdir tree is built
        self.build_tree()

    def build_tree(self) -> None:
        '''Build the main directory and tree '''
        self.path.mkdir(exist_ok=True)
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        self.checkpoint_path.touch() # must be done AFTER subdirectory creation, since the checkpoint file resides in the "checkpoint" subdirectory
        LOGGER.debug(f'Built filetree for {self.mol_name}')

    def __repr__(self) -> str:
        disp_attrs = (
            'mol_name',
            'base_mol_name',
            'path',
            'exclusion',
            'solvent'
        )
        attr_str = ', '.join(f'{attr}={getattr(self, attr)}' for attr in disp_attrs)

        return f'Polymer({attr_str})'
    
    def empty(self) -> None:
        '''Undoes build_tree - intended to "reset" a directory - NOTE : will break most functionality if build_tree() is not subsequently called'''
        filetree.clear_dir(self.path)
        LOGGER.warning(f'Cleared all contents of {self.mol_name}')

# ATTRIBUTE TRANSFER AND CLONING
    def transfer_file_attr(self, attr_name : str, other : 'Polymer') -> None:
        '''Copy a file at a particular attribute, modify its path, and set the appropriate attribute in the receiving Polymer'''
        file_path = getattr(self, attr_name)
        assert(isinstance(file_path, Path))

        subdir_name = self.subdirs_reverse[file_path.parent]
        curr_subdir = getattr(self, subdir_name)
        new_subdir = getattr(other, subdir_name)

        new_file_path = filetree.exchange_parent(file_path, old_parent=curr_subdir, new_parent=new_subdir) # create a new path with the destination parent tree
        new_file_path = new_file_path.with_name(new_file_path.name.replace(self.mol_name, other.mol_name)) # replace any references to the original file in the files name

        copyfile(file_path, new_file_path)
        setattr(other, attr_name, new_file_path)
        LOGGER.info(f'Transferred file ref at {self.mol_name}.{attr_name} to {new_file_path}')

    def transfer_attr(self, attr_name : str, other : 'Polymer') -> None:
        '''Copy an attributes value to another Polymer instance'''
        attr = getattr(self, attr_name)
        if isinstance(attr, Path):
            self.transfer_file_attr(attr_name, other)
        else:
            setattr(other, attr_name, copy.copy(attr)) # create copy of value to avoid shared references
            LOGGER.info(f'Copied attribute {attr_name} from {self.mol_name} to {other.mol_name}')

    def clone(self, dest_dir : Optional[Path]=None, clone_name : Optional[str]=None, exclusion : Optional[float]=None, clone_solvent : bool=True,
              clone_structures : bool=False, clone_monomers : bool=True, clone_ff : bool=True, clone_charges : bool=False, clone_sims : bool=False) -> 'Polymer':
        '''
        Create a copy of a Polymer in a specified directory with a new name, updating file references accordingly
        If neither a new Path nor name are specified, will default to cloning in the same parent directory with "_clone" affixed to self's name
        Will raise PermissionError if attempting to clone over the original Polymer
        '''
        # checking for defaults
        if dest_dir is None:
            dest_dir = self.parent_dir

        if not dest_dir.exists():
            raise FileNotFoundError(f'Clone destination directory {dest_dir} does not exist')

        if clone_name is None:
            clone_name = f'{self.mol_name}_clone'

        if exclusion is None:
            exclusion = self.exclusion
        
        if (dest_dir == self.parent_dir) and (clone_name == self.mol_name):
            raise PermissionError(f'Cannot clone Polymer at {self.path} over itself')

        # creating the clone
        LOGGER.info(f'Creating clone of {self.mol_name}')
        clone = Polymer(parent_dir=dest_dir, mol_name=clone_name, exclusion=exclusion)
        clone.base_mol_name = self.mol_name # keep record of identity of original molecule

        if clone_solvent:
            self.transfer_attr('solvent', clone)

        if clone_structures:
            self.transfer_attr('structure_file', clone)

        if clone_monomers:
            self.transfer_attr('monomer_file', clone)
            self.transfer_attr('monomer_file_chgd', clone)

        if clone_ff:
            self.transfer_attr('ff_file', clone)

        if clone_charges:
            self.transfer_attr('charges', clone)
            # TODO : implement transfer of SDF file dict (need more flexible implementation of transfer_file_attr)

        if clone_sims:
            copytree(self.MD, clone.MD, dirs_exist_ok=True)
            
        clone.to_file() # ensure changes are reflected in clone;s checkpoint
        LOGGER.info(f'Successfully created clone of {self.mol_name} at {clone.path}')

        return clone

# CHECKPOINT FILE
    def to_file(self) -> None:
        '''Save directory object to disc - for checkpointing and non-volatility'''
        LOGGER.debug(f'Updating checkpoint file of {self.mol_name}')
        with self.checkpoint_path.open('wb') as checkpoint_file:
            pickle.dump(self, checkpoint_file)

    @classmethod
    def from_file(cls, checkpoint_path : Path) -> 'Polymer':
        '''Load a saved directory tree object from disc'''
        assert(checkpoint_path.suffix == '.pkl')

        LOGGER.debug(f'Loading Polymer from {checkpoint_path}')
        with checkpoint_path.open('rb') as checkpoint_file:
            return pickle.load(checkpoint_file)
        
    @staticmethod
    def update_checkpoint(funct : Callable) -> Callable[[Any], Optional[Any]]: # NOTE : this deliberately doesn't have a "self" arg!
        '''Decorator for updating the on-disc checkpoint file after a function updates a Polymer attribute'''
        def update_fn(self, *args, **kwargs) -> Optional[Any]:
            ret_val = funct(self, *args, **kwargs) # need temporary value so update call can be made before returning
            self.to_file()
            return ret_val
        return update_fn
        
# STRUCTURE FILE PROPERTIES
    @property
    def has_structure_data(self) -> bool:
        return (self.structure_file is not None)

    @property
    def has_monomer_data_uncharged(self) -> bool:
        return (self.monomer_file is not None)
    
    @property
    def has_monomer_data_charged(self) -> bool:
        return (self.monomer_file_chgd is not None)

    @property
    def has_monomer_data(self) -> bool:
        '''Checking if ANY monomer file is present (charged or not)'''
        # return (self.monomer_file_ranked is not None)
        # return any(self.monomer_file, self.monomer_file_chgd)
        return (self.has_monomer_data_uncharged or self.has_monomer_data_charged)
    
    @property
    def monomer_file_ranked(self) -> Path:
        '''Choose monomer file from those available according to a ranked priority - returns Nonetype if no files are specified'''
        MONO_PRIORITY_ORDER = ('monomer_file_chgd', 'monomer_file') # NOTE : order here isn't arbitrary, establishes FIFO priority for monomer files - must match named PolymerInfo attributes
        for mono_file_type in MONO_PRIORITY_ORDER:
            if (possible_monofile := getattr(self, mono_file_type)) is not None:
                return possible_monofile
        else:
            return None

    @property
    def monomer_data_uncharged(self) -> dict[str, Any]:
        '''Load monomer information from file'''
        if not self.has_monomer_data_uncharged:
            raise FileExistsError(f'No monomer file exists for {self.mol_name}')

        with self.monomer_file.open('r') as json_file: 
            return json.load(json_file, object_hook=unserialize_monomer_json)
    monomer_data = monomer_data_uncharged # alias for backwards compatibility
    
    @property
    def monomer_data_charged(self) -> dict[str, Any]:
        '''Load monomer information with charges from file'''
        if not self.has_monomer_data_charged:
            raise FileExistsError(f'No monomer file with charges exists for {self.mol_name}')

        with self.monomer_file_chgd.open('r') as json_file: 
            return json.load(json_file, object_hook=unserialize_monomer_json)
        
# RDKit PROPERTIES 
    @property
    def rdmol_topology(self) -> RDMol:
        '''Load an RDKit Molecule object directly from structure file'''
        return Chem.MolFromPDBFile(str(self.structure_file), removeHs=False)
    
    @property
    def largest_rdmol(self) -> RDMol:
        '''Return the largest sub-molecule in the structure file Topology. Intended to differentiate target molecules from solvent'''
        return max(Chem.rdmolops.GetMolFrags(self.rdmol_topology, asMols=True), key=lambda mol : mol.GetNumAtoms())
    rdmol = largest_rdmol # alias for simplicity

    @property
    def n_atoms(self):
        '''Number of atoms in the main polymer chain (excluding solvent)'''
        return self.rdmol.GetNumAtoms()

    @property
    def mol_bbox(self):
        '''Return the bounding box size (in angstroms) of the molecule represented'''
        return self.rdmol.GetConformer().GetPositions().ptp(axis=0) * angstrom
    
    @property
    def box_vectors(self):
        '''Dimensions fo the periodic simulation box'''
        return self.mol_bbox + 2*self.exclusion # 2x accounts for the fact that exclusion must occur on either side of the tight bounding box
    
    @property
    def box_volume(self):
        return general.product(self.box_vectors)

# OpenFF MOLECULE PROPERTIES
    def off_topology_matched(self, strict : bool=True, verbose : bool=False, chgd_monomers : bool=False, topo_only : bool=True) -> tuple[Topology, Optional[list[SubstructSummary]], Optional[bool]]:
        '''Performs a monomer substructure match and returns the resulting OpenFF Topology'''
        monomer_path = self.monomer_file_chgd if chgd_monomers else self.monomer_file
        assert(self.structure_file.exists())
        assert(monomer_path.exists())

        LOGGER.info('Loading OpenFF Topology WITH monomer graph match')
        off_topology, substructs, error_state = Topology.from_pdb_and_monomer_info(str(self.structure_file), monomer_path, strict=strict, verbose=verbose)

        if topo_only:
            return off_topology
        return off_topology, substructs, error_state 
    
    def largest_offmol_matched(self, *topo_args, **topo_kwargs) -> Molecule: 
        '''Return the largest sub-molecule in the structure file Topology. Intended to differentiate target molecules from solvent'''
        topo = self.off_topology_matched(*topo_args, **topo_kwargs)
        if not isinstance(topo, Topology):
            topo = topo[0] # handles the case where topo_only is False - TODO : make this less terrible

        return max(topo.molecules, key=lambda mol : mol.n_atoms)
    offmol_matched = largest_offmol_matched # alias for simplicity
    
    # Property wrappers for default cases - simplifies notation for common usage. Rematched versions are still exposed thru *_matched OFF methods
    @property
    def off_topology(self):
        '''Cached version of graph-matched Topology to reduce load times'''
        if self._off_topology is None:
            self._off_topology = self.off_topology_matched() # cache topology match
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._off_topology
    
    @property
    def off_topology_unmatched(self):
        '''Cached version of Topology which is explicitly NOT graph-matched (limited chemical info but compatible with vanilla OpenFF)'''
        if self._off_topology_unmatched is None:
            assert(self.structure_file.exists())
            LOGGER.info('Loading OpenFF Topology WITHOUT monomer graph match')
            
            raw_offmol = Molecule.from_file(self.structure_file)
            frags = Chem.rdmolops.GetMolFrags(raw_offmol.to_rdkit(), asMols=True) # fragment Molecule in the case that multiple true molecules are present
            self._off_topology_unmatched = Topology.from_molecules(Molecule.from_rdkit(frag) for frag in frags)

        return self._off_topology_unmatched 

    @property 
    def largest_offmol(self):
        '''Cached version of primary Molecule (WITH graph match) to reduce load times'''
        if self._offmol is None:
            self._offmol = self.offmol_matched()
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._offmol
    offmol = largest_offmol # alias for simplicity
    
    @property 
    def largest_offmol_unmatched(self):
        '''Cached version of primary Molecule (WITHOUT graph match) to reduce load times'''
        if self._offmol_unmatched is None:
            self._offmol_unmatched = max(self.off_topology_unmatched.molecules, key=lambda mol : mol.n_atoms)
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._offmol_unmatched
    offmol_unmatched = largest_offmol_unmatched # alias for simplicity

# OpenFF FORCEFIELD PROPERTIES
    @update_checkpoint
    def create_FF_file(self, xml_src : Path, return_lib_chgs : bool=False) -> tuple[ForceField, Optional[list[LibraryChargeHandler]]]:
        '''Generate an OFF force field with molecule-specific (and solvent specific, if applicable) Library Charges appended'''
        if not self.has_monomer_data_uncharged:
            raise MissingChargedMonomerData

        ff_path = self.FF/f'{self.mol_name}.offxml' # path to output library charges to
        forcefield, lib_chgs = write_lib_chgs_from_mono_data(self.monomer_data_charged, xml_src, output_path=ff_path)

        if self.solvent is not None:
            forcefield = ForceField(ff_path, self.solvent.forcefield_file, allow_cosmetic_attributes=True) # use both the polymer-specific xml and the solvent FF xml to make hybrid forcefield
            forcefield.to_file(ff_path)
            LOGGER.info(f'Detected solvent "{self.solvent.name}", merged solvent and molecule force field')

        self.ff_file = ff_path # ensure change is reflected in directory info
        LOGGER.info(f'Successfully generated forcefield file for "{self.mol_name}"')

        if return_lib_chgs:
            return forcefield, lib_chgs
        return forcefield
    
    @property
    def forcefield(self):
        if self.ff_file is None:
            raise MissingForceFieldData

        return ForceField(self.ff_file, allow_cosmetic_attributes=True)
    
# CHARGING
    @update_checkpoint
    def register_charges(self, charge_method : str, charges : ndarray[float]) -> None:
        '''For inserting/overwriting available charge sets in local session and on file'''
        assert(general.hasunits(charges)) # ensure charges have units (can't assign to Molecules if not)
        LOGGER.debug(f'Registering charges from {charge_method} to {self.mol_name}')
        self.charges[charge_method] = charges

    @update_checkpoint
    def unregister_charges(self, charge_method) -> None:
        LOGGER.debug(f'Unregistering charges from {charge_method} to {self.mol_name}')
        '''For removing available charge sets in local session and on file'''
        self.charges.pop(charge_method, None) # TOSELF : can get rid of None default arg if KeyError is istaed desired

    @update_checkpoint
    def assign_charges_by_lookup(self, charge_method : str) -> None:
        '''Choose which registered charges should be applied to the cached Molecule by key lookup'''
        LOGGER.info(f'Assigning charges from {charge_method} to {self.mol_name}\'s OpenFF Molecule')
        self.offmol.partial_charges = self.charges[charge_method] # NOTE : explicitly using property version of offmol (i.e. not _offmol) in case Molecule isn't yet instantiated
        self.charge_method = charge_method 
    
    def assign_charges(self, charge_method : str, charges : ndarray[float]) -> None:
        '''Wrap registration and assignment for convenience'''
        self.register_charges(charge_method, charges) # doesn't need "update_checkpoint" flag, as all internal methods already do so
        self.assign_charges_by_lookup(charge_method)

    def has_charges_for(self, charge_method : str) -> bool:
        '''Return whether or not all requisite charge info (i.e. a list of charges and a charged .SDF) is present for the soecified charging method'''
        return all(
            charge_method in reg_dict # simple key-based check in charge set and charge file registries
                for reg_dict in (self.charges, self.structure_files_chgd)
        )
    
    @update_checkpoint
    def charge_and_save_molecule(self, charger : MolCharger, *topo_args, **topo_kwargs) -> tuple[Molecule, Path]:
        '''Generates and registers 1) an .SDF file for the charged molecule and 2) an charge entry for just the Molecule's charges'''
        charge_method = charger.METHOD_NAME

        unchgd_mol = self.offmol_matched(*topo_args, **topo_kwargs) # load a fresh, uncharged molecule from graph match (important to avoid charge contamination)
        chgd_mol = charger.charge_molecule(unchgd_mol)
        self.register_charges(charge_method, chgd_mol.partial_charges)
        LOGGER.info(f'Registered "{charge_method}" charges to "{self.mol_name}"')

        sdf_path = self.SDF/f'{self.mol_name}_charged_{charge_method}.sdf'
        chgd_mol.properties['metadata'] = [atom.metadata for atom in chgd_mol.atoms] # need to store metadata as separate property, since SDF does not preserved metadata atomwise
        chgd_mol.to_file(str(sdf_path), file_format='SDF')
        LOGGER.info(f'Wrote {self.mol_name} Molecule with {charge_method} charges to sdf file')
        self.structure_files_chgd[charge_method] = sdf_path

        return chgd_mol, sdf_path
    
    def charged_offmol_from_sdf(self, charge_method : str) -> Molecule:
        '''Loads a charged Molecule from a registered charge method by key lookup
        Raises KeyError if the method requested is not registered'''
        return load_matched_charged_molecule(self.structure_files_chgd[charge_method], assume_ordered=True)

    def assert_charges_for(self, charger : MolCharger, strict : bool=True, verbose : bool=False) -> Molecule:
        '''Return charged molecule associated with a particular charger's method
        If not already extant, will generate new charge sets and SDFs'''
        charge_method = charger.METHOD_NAME

        if self.has_charges_for(charge_method): # if charges and charge Molecule SDFs already exist for the current method
            LOGGER.info(f'Found existing "{charge_method}" partial charges for {self.mol_name}')
            cmol = self.charged_offmol_from_sdf(charge_method)
        else:
            LOGGER.warning(f'Found no existing "{charge_method}" partial charges for {self.mol_name}')
            cmol, sdf_path = self.charge_and_save_molecule(charger, strict=strict, verbose=verbose, chgd_monomers=False, topo_only=True) # ensure only uncharged monomers are used to avoid charge contamination

        return cmol
    
    def compare_charges(self, charge_method_1 : str, charge_method_2 : str, *args, **kwargs) -> tuple[Figure, Axes]:
        '''Plot a heat map showing the atomwise discrepancies in partial charges between any pair of registered charge sets'''
        chgd_rdmol1 = self.charged_offmol_from_sdf(charge_method_1).to_rdkit()
        chgd_rdmol2 = self.charged_offmol_from_sdf(charge_method_2).to_rdkit()

        return compare_chgd_rdmols(chgd_rdmol1, chgd_rdmol2, charge_method_1, charge_method_2, *args, **kwargs)
    
    @update_checkpoint
    def create_charged_monomer_file(self, residue_charges : ResidueChargeMap):
        '''Create a new copy of the current monomer file with residue-wise mapping of substructure id to charges'''
        if not self.has_monomer_data_uncharged: # specifically checking for uncharged monomers (hence why self.has_monomers is not used here)
            raise MissingMonomerData

        if self.solvent is not None and self.solvent.charges is not None: # ensure solvent "monomer" charge entries are also recorded
            residue_charges = {**residue_charges, **self.solvent.monomer_json_data['charges']} 
        chgd_monomer_data = {**self.monomer_data_uncharged} # load uncharged monomer_data and create copy - TOSELF : this variable name is confusing but is correct (will be charged subsequently)
        chgd_monomer_data['charges'] = residue_charges

        chgd_mono_path = self.monomer_file.with_name(f'{self.mol_name}_charged.json')
        with chgd_mono_path.open('w') as chgd_json:
            json.dump(chgd_monomer_data, chgd_json, indent=4)
        LOGGER.info(f'Created new monomer file for "{self.mol_name}" with monomer-averaged charge entries')

        self.monomer_file_chgd = chgd_mono_path # record path to new json file

# FILE POPULATION AND MANAGEMENT
    @staticmethod
    def _file_population_factory(file_name_affix : str, subdir_name : str, targ_attr : str, desc : str='') -> Callable[[Path, bool], None]:
        '''Factory method for creating functions to populate files from external folders
        TODO - find more elegant way to inject object-dependent attributes, figure out how to use update_checkpoints() inside of local namespace'''
        if desc:
            desc += ' ' # add space after non-empty descriptors for legibility
        
        def population_method(self, source_dir : Path, assert_exists : bool=False) -> None:
            '''Boilerplate for file population'''
            file_name = f'{self.mol_name}{file_name_affix}'
            LOGGER.info(f'Acquiring {desc}file(s) {file_name} from {source_dir}')
            src_path = source_dir/file_name

            if not src_path.exists():
                if assert_exists: 
                    raise FileNotFoundError(f'No file {src_path} exists') # raise an error for missing files if explicitly told to...
                else:
                    LOGGER.warning(f'No file {src_path} exists') # ... otherwise, record that this file is missing
            else:
                dest_path = getattr(self, subdir_name)/file_name
                copyfile(src_path, dest_path)
                setattr(self, targ_attr, dest_path)

        return population_method

    # TOSELF : need to explicitly call update_checkpoint (without decorator syntax) due to annoying internal namespace restrictions
    populate_pdb = update_checkpoint(_file_population_factory(file_name_affix='.pdb', subdir_name='structures', targ_attr='structure_file', desc='structure'))
    populate_monomers= update_checkpoint(_file_population_factory(file_name_affix='.json', subdir_name='monomers', targ_attr='monomer_file', desc='monomer'))

    def populate_mol_files(self, struct_dir : Path, monomer_dir : Path=None) -> None:
        '''
        Populates a Polymer with the relevant structural and monomer files from a shared source ("data dump") folder
        Assumes that all structure and monomer files will have the same name as the Polymer in question
        '''
        self.populate_pdb(struct_dir, assert_exists=True)
        
        if monomer_dir is None:
            monomer_dir = struct_dir # if separate monomer folder isn't explicitly specified, assume monomers are in structure file folder
        self.populate_monomers(monomer_dir, assert_exists=False) # don't force error if no monomers are found

# SOLVATION
    # doesn't need "update_checkpoint" decorator, as no own attrs are being updated
    def solvate(self, solvent : Solvent, template_path : Path, dest_dir : Path=None, exclusion : float=None, precision : int=4) -> 'Polymer':
        '''Create a clone of a Polymer and solvate it in a box defined by the polymer's bounding box + an exclusion buffer'''
        if not self.has_structure_data:
            raise MissingStructureData
        
        if self.solvent is not None:
            raise AlreadySolvatedError
        
        if exclusion is None:
            exclusion = self.exclusion

        LOGGER.info(f'Solvating "{self.mol_name}" in {solvent.name}')
        solvated_name = f'{self.mol_name}_solv_{solvent.name}'
        solva_dir = self.clone( # logging performed implicitly within cloning and solvating steps here
            dest_dir=dest_dir,
            clone_name=solvated_name,
            exclusion=exclusion,
            clone_solvent=False,
            clone_structures=True,
            clone_monomers=True,
            clone_ff=True,
            clone_charges=False,
            clone_sims=False
        ) 

        solva_dir.structure_file = packmol_solvate_wrapper( # generate and point to solvated PDB structure
            polymer_pdb = self.structure_file, 
            solvent_pdb = solvent.structure_file, 
            outdir      = solva_dir.structures, 
            outname     = solvated_name, 
            template_path = template_path,
            N           = round(solva_dir.box_volume * solvent.number_density),
            box_dims    = solva_dir.box_vectors,
            precision   = precision
        )

        if solva_dir.has_monomer_data: # this will be copied over during the cloning process (ensure "clone_monomers" flag is set)
            solvated_mono_data = {**solva_dir.monomer_data}   # note that this is explicitly UNCHARGED monomer data
            for field, values in solvated_mono_data.items():     # this merge strategy ensures solvent data does not overwrite or append extraneous data
                values.update(solvent.monomer_json_data[field])  # specifically, charges will not be written to an uncharged json (which would screw up graph match and load)

            with solva_dir.monomer_file.open('w') as solv_mono_file:
                json.dump(solvated_mono_data, solv_mono_file, indent=4)

        solva_dir.solvent = solvent # set this only AFTER solvated files have been created
        solva_dir.to_file() # ensure checkpoint file is created for newly solvated directory
        LOGGER.info('Successfully converged on solvent packing')

        return solva_dir

    @update_checkpoint
    def solvate_legacy(self, solvent : Solvent, template_path : Path, exclusion : float=None, precision : int=4) ->  'Polymer':
        '''Applies packmol solvation routine to an extant Polymer
        This is the original implementation of solvation, kept for debug reasons'''
        if not self.has_structure_data:
            raise MissingStructureData
        
        if self.solvent is not None:
            raise AlreadySolvatedError

        if exclusion is None:
            exclusion = self.exclusion # default to same exclusion as parent

        solvated_name = f'{self.mol_name}_solv_{solvent.name}'
        solva_dir = Polymer(parent_dir=self.parent_dir, mol_name=solvated_name)
        solva_dir.base_mol_name = self.mol_name # TOSELF : temporary, ensure that a solvated mol is stil keyable by the original molecule name

        solva_dir.exclusion = exclusion
        box_vectors = self.mol_bbox + 2*exclusion # can't use either dirs "box_vectors" property directly, since the solvated dir has no structure file yet and the old dir may have different exclusion
        V_box = general.product(box_vectors)

        LOGGER.info(f'Creating copy of {self.mol_name}, now solvated in {solvent.name}')
        solva_dir.structure_file = packmol_solvate_wrapper( # generate and point to solvated PDB structure
            polymer_pdb = self.structure_file, 
            solvent_pdb = solvent.structure_file, 
            outdir      = solva_dir.structures, 
            outname     = solvated_name, 
            template_path = template_path,
            N           = round(V_box * solvent.number_density),
            box_dims    = box_vectors,
            precision   = precision
        )

        if self.has_monomer_data:
            solvated_mono_data = {**self.monomer_data_uncharged}           # note that this is explicitly UNCHARGED monomer data
            for field, values in solvated_mono_data.items():     # this merge strategy ensures solvent data does not overwrite or append extraneous data
                values.update(solvent.monomer_json_data[field])  # specifically, charges will not be written to an uncharged json (which would screw up graph match and load)

            solva_mono_path = solva_dir.monomers/f'{solvated_name}.json'
            with solva_mono_path.open('w') as solv_mono_file:
                json.dump(solvated_mono_data, solv_mono_file, indent=4)
            solva_dir.monomer_file = solva_mono_path

        solva_dir.solvent = solvent # set this only AFTER solvated files have been created
        solva_dir.to_file() # ensure checkpoint file is created for newly solvated directory
        LOGGER.info('Successfully converged solvent packing')

        return solva_dir
    
# SIMULATION
    @property
    def completed_sims(self) -> list[Path]:
        '''Return paths to all extant, non-empty simulation subdirectories'''
        return [sim_dir for sim_dir in self.MD.iterdir() if not filetree.is_empty(sim_dir)]
    
    @property
    def chrono_sims(self) -> list[Path]:
        '''Return paths of all extant simulation subdirectories in chronological order of creation'''
        return sorted(self.completed_sims, key=lambda path : general.extract_time(path.stem))
    
    @property
    def oldest_sim_dir(self) -> Path:
        '''Return the least recent simulation subdir'''
        return self.chrono_sims[0]

    @property
    def newest_sim_dir(self) -> Path:
        '''Return the most recent simulation subdir'''
        return self.chrono_sims[-1]

    def make_sim_dir(self, affix : Optional[str]='') -> Path:
        '''Create a new timestamped simulation results directory'''
        sim_name = f'{affix}{"_" if affix else ""}{timestamp_now()}'
        sim_dir = self.MD/sim_name
        sim_dir.mkdir(exist_ok=False) # will raise FileExistsError in case of overlap
        LOGGER.info(f'Created new Simulation directory "{sim_name}"')

        return sim_dir
    
    def interchange(self, charge_method : str):
        '''Create an Interchange object for a SMIRNOFF force field using internal structural files'''
        off_topology = self.off_topology_matched() # self.off_topology
        off_topology.box_vectors = self.box_vectors.in_units_of(nanometer) # set box vector to allow for periodic simulation (will be non-periodic if polymer box vectors are unset i.e. NoneType)

        self.assign_charges_by_lookup(charge_method) # assign relevant charges prior to returning molecule
        cmol = self.offmol 
  
        LOGGER.info(f'Creating SMIRNOFF Interchange for "{self.mol_name}"')
        return Interchange.from_smirnoff(force_field=self.forcefield, topology=off_topology, charge_from_molecules=[cmol]) # package FF, topoplogy, and molecule charges together and ship out
        
    def run_simulation_NPT(self, sim_params : simulation.SimulationParameters) -> Path:
        '''Initialize and run an OpenMM simulation in the Isothermal-Isobaric ensemble with a predefined set of setup parameters'''
        sim_folder = self.make_sim_dir()

        interchange = self.interchange(sim_params.charge_method)
        integrator  = LangevinMiddleIntegrator(sim_params.temperature, sim_params.friction_coeff, sim_params.timestep)
        barostat    = MonteCarloBarostat(sim_params.pressure, sim_params.temperature, sim_params.barostat_freq)
        sim = simulation.create_simulation(interchange, integrator, forces=[barostat])

        self.simulation_paths[sim_folder] = simulation.run_simulation(sim, output_folder=sim_folder, output_name=self.mol_name, sim_params=sim_params) # run simulation and record paths to output files

        return sim_folder

    def purge_sims(self, really : bool=False) -> None:
        '''Empties all extant simulation folders - MAKE SURE YOU KNOW WHAT YOU'RE DOING HERE'''
        if not really:
            raise PermissionError('Please confirm that you really want to clear simulation directories (this can\'t be undone!)')
        
        if self.completed_sims: # only attempt removal and logging if simulations are actually present
            filetree.clear_dir(self.MD)
            filetree.clear_dir(self.logs)
            self.simulation_paths.clear()
            LOGGER.warning(f'Deleted all extant simulations for {self.mol_name}')

# Filtering Polymers by attributes
MolFilter = Callable[[Polymer], bool] # typing template for explicitness

def filter_factory_by_attr(attr_name : str, condition : Callable[[Any], bool]=bool, inclusive : bool=True) -> MolFilter:
    '''Factory function for creating MolFilters for arbitrary individual Polymer attributes

    Takes the name of the attribute to look-up and a truthy "condition" function to validate attribute values (by default just bool() i.e. property exists)
    Resulting filter will return Polymers for which the condition is met by default, or those for which it isn't if inclusive == False
    '''
    def _filter(polymer : Polymer) -> bool:
        '''The actual case-specific filter '''
        attr_val = getattr(polymer,  attr_name)
        satisfied = condition(attr_val)
        
        if inclusive:
            return satisfied
        return not satisfied # invert truthiness if non-inclusive

    return _filter

has_monomers = lambda polymer : polymer.has_monomer_data
AM1_sized    = lambda polymer : 0 < polymer.n_atoms <= 300
# has_monomers = filter_factory_by_attr(attr_name='has_monomer_data')
# AM1_sized    = filter_factory_by_attr(attr_name='n_atoms', condition=lambda n : 0 < n <= 300)

# Manager class for collections of Polymers
class PolymerManager:
    '''Class for organizing, loading, and manipulating collections of Polymer objects'''
    def __init__(self, collection_dir : Path):
        self.collection_dir : Path = collection_dir
        self.log_dir        : Path = self.collection_dir/'Logs'
        self.polymers_list  : list[Polymer] = []

        self.collection_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.update_collection() # populate currently extant dirs

    def __repr__(self) -> str:
        disp_attrs = (
            'collection_dir',
            'log_dir'
        )
        attr_str = ', '.join(f'{attr}={getattr(self, attr)}' for attr in disp_attrs)

        return f'PolymerManager({attr_str})'

# READING EXISTING Polymers
    def update_collection(self, return_missing : bool=False) -> Optional[list[Path]]:
        '''
        Load existing polymer directories from target directory into memory
        Return a list of the checkpoint patfhs which contain faulty/no data
        '''
        LOGGER.debug(f'Refreshing available Polymers available in PolymerManager at {self.collection_dir}')
        found_polymers, missing_chk_data = [], []
        for checkpoint_file in self.collection_dir.glob('**/*_checkpoint.pkl'):
            try:
                found_polymers.append(Polymer.from_file(checkpoint_file))
            except EOFError:
                missing_chk_data.append(checkpoint_file)
                # print(f'Missing checkpoint file info in {polymer}') # TODO : find better way to log this
        self.polymers_list = found_polymers # avoiding direct append ensures no double-counting if run multiple times (starts with empty list each time)

        if return_missing:
            return missing_chk_data
        
    @staticmethod
    def refresh_listed_dirs(funct : Callable) -> Callable[[Any], Optional[Any]]: # NOTE : this deliberately doesn't have a "self" arg!
        '''Decorator for updating the internal list of Polymers after a function'''
        def update_fn(self, *args, **kwargs) -> Optional[Any]:
            ret_val = funct(self, *args, **kwargs) # need temporary value so update call can be made before returning
            self.update_collection(return_missing=False)
            return ret_val
        return update_fn
    
# PROPERTIES
    @property
    def n_mols(self) -> int:
        '''Number of Polymers currently being tracked'''
        return len(self.polymers_list)

    @property
    def polymers_list_by_size(self) -> list[Polymer]:
        '''Return all polymer directories in collection, ordered by molecule size'''
        return sorted(self.polymers_list, key=lambda mdir : mdir.n_atoms)

    @property
    def polymers(self) -> dict[str, Polymer]:
        '''Currently extant Polymers, keyed by their molecule name (convenient for applications where frequent name lookup/str manipulation is required)'''
        return {polymer.mol_name : polymer for polymer in self.polymers_list} # NOTE : formulated as a property in order to generate unique copies of the dict when associating to another variable (prevents shared overwrites)
    
    @property
    def all_completed_sims(self) -> dict[str, list[Path]]: 
        '''Get a dict of currently populated simulation folders itermized by molecule name'''
        return {
            mol_name : polymer.completed_sims
                for mol_name, polymer in self.polymers.items()
                    if polymer.completed_sims # don't report directories without simulations
        }

    def filtered_by(self, filters : Union[MolFilter, Iterable[MolFilter]]) -> dict[str, Polymer]:
        '''Return name-keyed dict of all Polymers in collection which meet all of a set of filtering conditions'''
        if not isinstance(filters, Iterable):
            filters = [filters] # handle the singleton case

        return {
            polymer.mol_name : polymer
                for polymer in self.polymers_list
                    if all(_filter(polymer) for _filter in filters)
        }

# FILE AND Polymer GENERATION
    @refresh_listed_dirs
    def populate_collection(self, struct_dir : Path, monomer_dir : Path=None) -> None:
        '''Generate Polymers via files extracted from a lumped PDB/JSON directory'''
        for pdb_path in struct_dir.glob('**/*.pdb'):
            mol_name = pdb_path.stem
            parent_dir = self.collection_dir/mol_name
            parent_dir.mkdir(exist_ok=True)

            polymer = Polymer(parent_dir, mol_name)
            polymer.populate_mol_files(struct_dir=struct_dir, monomer_dir=monomer_dir)

    @refresh_listed_dirs
    def solvate_collection(self, solvents : Iterable[Solvent], template_path : Path, exclusion : float=None) -> None:
        '''Make solvated versions of all listed Polymers, according to a collection of solvents'''      
        for solvent in solvents:
            if solvent is None: # handle the case where a null solvent is passed (technically a valid Solvent for typing reasons)
                continue

            for polymer in self.polymers_list:
                if (polymer.solvent is None): # don't double-solvate any already-solvated systems
                    solv_dir = polymer.solvate(solvent=solvent, template_path=template_path, exclusion=exclusion)
                    self.polymers_list.append(solv_dir) # ultimately pointless, as this will be cleared and reloaded upon refresh. Still, nice to be complete :)

# PURGING (DELETION) METHODS
    def purge_logs(self, really : bool=False) -> None: # NOTE : deliberately undecorated, as no Polymer reload is required
        if not really:
            raise PermissionError('Please confirm that you really want to clear all directories (this can\'t be undone!)')
        filetree.clear_dir(self.log_dir)
        LOGGER.warning(f'Deleted all log files at {self.log_dir}')

    @refresh_listed_dirs
    def purge_collection(self, really : bool=False, purge_logs: bool=False) -> None:
        if not really:
            raise PermissionError('Please confirm that you really want to clear all directories (this can\'t be undone!)')
        
        for subdir in self.collection_dir.iterdir():
            if subdir != self.log_dir:
                filetree.clear_dir(subdir)
                subdir.rmdir() # clean up now-emptied subdirectories
        LOGGER.warning(f'Deleted all Polymers found in {self.collection_dir}')
        
        if purge_logs:
            self.purge_logs(really=really)
    
    @refresh_listed_dirs
    def purge_sims(self, really : bool=False) -> None:
        if not really:
            raise PermissionError('Please confirm that you really want to clear all directories (this can\'t be undone!)')
        
        for polymer in self.polymers_list:
            polymer.purge_sims(really=really)