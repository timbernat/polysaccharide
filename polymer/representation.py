# Custom imports
from .. import general, filetree
from ..solvation.packmol import packmol_solvate_wrapper

from ..charging.application import load_matched_charged_molecule

from ..simulation.records import SimulationPaths, SimulationParameters

from ..analysis.trajectory import load_traj
from ..molutils.rdmol.rdcompare import compare_chgd_rdmols
from ..molutils.rdmol.rdprops import assign_ordered_atom_map_nums

from .exceptions import *
from .monomer import MonomerInfo

# Typing and subclassing
from typing import Any, Callable, ClassVar, Iterable, Optional, TypeAlias, Union
from mdtraj import Trajectory
from numpy import ndarray
from matplotlib.pyplot import Figure, Axes

from ..charging.application import MolCharger
from ..solvation.solvent import Solvent
from ..extratypes import SubstructSummary, ResidueChargeMap, RDMol
SimDirFilter : TypeAlias = Callable[[SimulationPaths, SimulationParameters], bool]

# File I/O
import copy
from pathlib import Path
from shutil import copyfile, copytree
import pickle

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# Cheminformatics and Molecular Dynamics
from rdkit import Chem

from openff.toolkit import ForceField
from openff.toolkit.topology import Topology, Molecule

from openmm.unit import angstrom, nanometer


# Helper functions
def create_subdir_properties(cls):
    '''For dynamically adding all subdirectories as accessible property fields - cleans up namespace for Polymer'''
    for dir_name in cls._SUBDIRS:
        def make_prop(subdir_name : str):
            return property(fget=lambda self : self.subdirs[subdir_name])
        
        setattr(cls, dir_name, make_prop(dir_name))
    return cls

# Polymer representation class
@create_subdir_properties
class Polymer:
    '''For representing standard directory structure and requisite information for polymer structures, force fields, and simulations'''
    _SUBDIRS : ClassVar[tuple[str, ...]] = ( # directories with these names will be present in all polymer directories by standard
        'structures',
        'monomers',
        'SDF',
        'MD',
        'checkpoint',
        'logs'
    )

    DATE_FMT : ClassVar[general.Timestamp] = general.Timestamp() # can set this class variable to alter date formatting

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
        
        self.monomer_file_uncharged : Optional[Path]  = None # .JSON
        self.monomer_file_charged   : Optional[Path]  = None # .JSON
        self.structure_file         : Optional[Path]  = None # .PDB
        self.structure_files_chgd   : dict[str, Path] = {} # dict of .SDF

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
        LOGGER.info(f'Built filetree for {self.mol_name}')

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
        other.to_file() # ensure newly set attributes are reflected in checkpoint
        LOGGER.info(f'Transferred file ref at {self.mol_name}.{attr_name} to {new_file_path}')

    def transfer_attr(self, attr_name : str, other : 'Polymer') -> None:
        '''Copy an attributes value to another Polymer instance'''
        attr = getattr(self, attr_name)
        if isinstance(attr, Path):
            self.transfer_file_attr(attr_name, other)
        else:
            setattr(other, attr_name, copy.copy(attr)) # create copy of value to avoid shared references
            other.to_file() # ensure newly set attributes are reflected in checkpoint
            LOGGER.info(f'Copied attribute {attr_name} from {self.mol_name} to {other.mol_name}')

    def clone(self, dest_dir : Optional[Path]=None, clone_affix : str='clone', exclusion : Optional[float]=None, clone_solvent : bool=True,
              clone_structures : bool=False, clone_monomers : bool=True, clone_charges : bool=False, clone_sims : bool=False) -> 'Polymer':
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

        clone_name = f'{self.mol_name}{"_" if clone_affix else ""}{clone_affix}'
        if (dest_dir == self.parent_dir) and (clone_name == self.mol_name):
            raise PermissionError(f'Cannot clone Polymer at {self.path} over itself')

        if exclusion is None:
            exclusion = self.exclusion
        
        # creating the clone
        LOGGER.info(f'Creating clone of {self.mol_name}')
        clone = Polymer(parent_dir=dest_dir, mol_name=clone_name, exclusion=exclusion)
        clone.base_mol_name = self.base_mol_name # keep record of identity of original molecule

        if clone_solvent:
            self.transfer_attr('solvent', clone)

        if clone_structures:
            self.transfer_attr('structure_file', clone)

        if clone_monomers:
            self.transfer_attr('monomer_file_uncharged', clone)
            self.transfer_attr('monomer_file_charged', clone)

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
    def monomer_file(self) -> Path:
        '''Alias for backwards-compatibility reasons'''
        return self.monomer_file_uncharged
    
    @monomer_file.setter
    def monomer_file(self, mono_path : Path) -> None:
        '''Allow for mutability via alias'''
        self.monomer_file_uncharged = mono_path

    @property
    def has_monomer_info_uncharged(self) -> bool:
        return (self.monomer_file_uncharged is not None)
    
    @property
    def has_monomer_info_charged(self) -> bool:
        return (self.monomer_file_charged is not None)

    @property
    def has_monomer_info(self) -> bool:
        '''Checking if ANY monomer file is present (charged or not)'''
        # return any(self.monomer_file_uncharged, self.monomer_file_charged)
        return (self.has_monomer_info_uncharged or self.has_monomer_info_charged)

    @property
    def monomer_info_uncharged(self) -> MonomerInfo:
        '''Load monomer information from file'''
        if not self.has_monomer_info_uncharged:
            raise MissingMonomerDataUncharged(f'No monomer file exists for {self.mol_name}')

        return MonomerInfo.from_file(self.monomer_file_uncharged)
    monomer_info = monomer_info_uncharged # alias for backwards compatibility
    
    @property
    def monomer_info_charged(self) -> MonomerInfo:
        '''Load monomer information with charges from file'''
        if not self.has_monomer_info_charged:
            raise MissingMonomerDataCharged(f'No monomer file with charges exists for {self.mol_name}')

        return MonomerInfo.from_file(self.monomer_file_charged)

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
    def n_atoms(self) -> int:
        '''Number of atoms in the main polymer chain (excluding solvent)'''
        return self.rdmol.GetNumAtoms()

    @property
    def mol_bbox(self) -> ndarray:
        '''Return the bounding box size (in angstroms) of the molecule represented'''
        return self.rdmol.GetConformer().GetPositions().ptp(axis=0) * angstrom
    
    @property
    def box_vectors(self) -> ndarray:
        '''Dimensions fo the periodic simulation box'''
        return self.mol_bbox + 2*self.exclusion # 2x accounts for the fact that exclusion must occur on either side of the tight bounding box
    
    @property
    def box_volume(self) -> float:
        return general.product(self.box_vectors)

# OpenFF MOLECULE PROPERTIES
    def off_topology_matched(self, strict : bool=True, verbose : bool=False, chgd_monomers : bool=False, topo_only : bool=True) -> tuple[Topology, Optional[list[SubstructSummary]], Optional[bool]]:
        '''Performs a monomer substructure match and returns the resulting OpenFF Topology'''
        monomer_path = self.monomer_file_charged if chgd_monomers else self.monomer_file_uncharged
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
    def off_topology(self) -> Topology:
        '''Cached version of graph-matched Topology to reduce load times'''
        if self._off_topology is None:
            self._off_topology = self.off_topology_matched() # cache topology match
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._off_topology
    
    @property
    def off_topology_unmatched(self) -> Topology:
        '''Cached version of Topology which is explicitly NOT graph-matched (limited chemical info but compatible with vanilla OpenFF)'''
        if self._off_topology_unmatched is None:
            assert(self.structure_file.exists())
            LOGGER.info('Loading OpenFF Topology WITHOUT monomer graph match')
            
            raw_offmol = Molecule.from_file(self.structure_file)
            frags = Chem.rdmolops.GetMolFrags(raw_offmol.to_rdkit(), asMols=True) # fragment Molecule in the case that multiple true molecules are present
            self._off_topology_unmatched = Topology.from_molecules(Molecule.from_rdkit(frag) for frag in frags)

        return self._off_topology_unmatched 

    @property 
    def largest_offmol(self) -> Molecule:
        '''Cached version of primary Molecule (WITH graph match) to reduce load times'''
        if self._offmol is None:
            self._offmol = self.offmol_matched()
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._offmol
    offmol = largest_offmol # alias for simplicity
    
    @property 
    def largest_offmol_unmatched(self) -> Molecule:
        '''Cached version of primary Molecule (WITHOUT graph match) to reduce load times'''
        if self._offmol_unmatched is None:
            self._offmol_unmatched = max(self.off_topology_unmatched.molecules, key=lambda mol : mol.n_atoms)
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._offmol_unmatched
    offmol_unmatched = largest_offmol_unmatched # alias for simplicity
    
# CHARGING
    @update_checkpoint
    def register_charges(self, charge_method : str, charges : ndarray[float]) -> None:
        '''For inserting/overwriting available charge sets in local session and on file'''
        assert(general.hasunits(charges)) # ensure charges have units (can't assign to Molecules if not)
        LOGGER.info(f'Registering charges from {charge_method} to {self.mol_name}')
        self.charges[charge_method] = charges

    @update_checkpoint
    def unregister_charges(self, charge_method) -> None:
        LOGGER.info(f'Unregistering charges from {charge_method} to {self.mol_name}')
        '''For removing available charge sets in local session and on file'''
        self.charges.pop(charge_method, None) # TOSELF : can get rid of None default arg if KeyError is istaed desired

    @update_checkpoint
    def assign_charges_by_lookup(self, charge_method : str) -> None:
        '''Choose which registered charges should be applied to the cached Molecule by key lookup'''
        LOGGER.info(f'Assigning precomputed {charge_method} charges to {self.mol_name}')
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
    charged_offmol = charged_offmol_from_sdf # alias for convenience

    @update_checkpoint
    def create_charged_monomer_file(self, residue_charges : ResidueChargeMap):
        '''Create a new copy of the current monomer file with residue-wise mapping of substructure id to charges'''
        LOGGER.warning('Generating new monomer JSON file with monomer-averaged charges')
        chgd_monomer_info = MonomerInfo( # create new monomer info object with the provided library charges
            monomers=self.monomer_info_uncharged.monomers,
            charges=residue_charges
        )

        if (self.solvent is not None) and (self.solvent.charges is not None): # ensure solvent "monomer" charge entries are also recorded for matching purposes
            chgd_monomer_info += self.solvent.monomer_info # inject molecule info from solvent

        chgd_mono_path = self.monomer_file_uncharged.with_name(f'{self.mol_name}_charged.json')
        chgd_monomer_info.to_file(chgd_mono_path) # write to new file for future reference
        self.monomer_file_charged = chgd_mono_path # record path to new json file only AFTER write is successful
        LOGGER.info(f'Generated new monomer file for "{self.mol_name}" with monomer-averaged charge entries')

    def assert_charges_for(self, charger : MolCharger, strict : bool=True, verbose : bool=False, return_cmol : bool=True) -> Optional[Molecule]:
        '''Return charged molecule associated with a particular charger's method
        If not already extant, will generate new charge sets and SDFs'''
        charge_method = charger.METHOD_NAME

        if self.has_charges_for(charge_method): # if charges and charge Molecule SDFs already exist for the current method
            LOGGER.info(f'Found existing "{charge_method}" partial charges for "{self.mol_name}"')
            cmol = self.charged_offmol_from_sdf(charge_method)
        else:
            LOGGER.warning(f'Found no existing "{charge_method}" partial charges for "{self.mol_name}"')
            cmol, sdf_path = self.charge_and_save_molecule(charger, strict=strict, verbose=verbose, chgd_monomers=False, topo_only=True) # ensure only uncharged monomers are used to avoid charge contamination

        if return_cmol:
            return cmol
    
    # Charge visualization
    def compare_charges(self, charge_method_1 : str, charge_method_2 : str, *args, **kwargs) -> tuple[Figure, Axes]:
        '''Plot a heat map showing the atomwise discrepancies in partial charges between any pair of registered charge sets'''
        chgd_offmol1 = self.charged_offmol_from_sdf(charge_method_1)
        chgd_offmol2 = self.charged_offmol_from_sdf(charge_method_2)
        chgd_rdmol1 = assign_ordered_atom_map_nums(chgd_offmol1.to_rdkit()) # ensure map numbers are present for correct matching
        chgd_rdmol2 = assign_ordered_atom_map_nums(chgd_offmol2.to_rdkit()) # ensure map numbers are present for correct matching

        return compare_chgd_rdmols(chgd_rdmol1, chgd_rdmol2, charge_method_1, charge_method_2, *args, **kwargs)
    
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
    populate_pdb      = update_checkpoint(_file_population_factory(file_name_affix='.pdb' , subdir_name='structures', targ_attr='structure_file', desc='structure'))
    populate_monomers = update_checkpoint(_file_population_factory(file_name_affix='.json', subdir_name='monomers'  , targ_attr='monomer_file_uncharged', desc='monomer'))

    def populate_mol_files(self, struct_dir : Path, monomer_dir : Optional[Path]=None) -> None:
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
    def _solvate(self, solvent : Solvent, template_path : Path, dest_dir : Optional[Path]=None, exclusion : Optional[float]=None, precision : int=4) -> None:
        '''Internal implementation for solvating with a single solvent via cloning and structure/Solvent reassignment'''
        if not self.has_structure_data:
            raise MissingStructureData
        
        if self.solvent is not None:
            raise AlreadySolvatedError
        
        if exclusion is None:
            exclusion = self.exclusion

        LOGGER.info(f'Solvating "{self.mol_name}" in {solvent.name}')
        solva_dir = self.clone( # logging performed implicitly within cloning and solvating steps here
            dest_dir=dest_dir,
            clone_affix=f'solv_{solvent.name}',
            exclusion=exclusion,
            clone_solvent=False,
            clone_structures=True,
            clone_monomers=True,
            clone_charges=False,
            clone_sims=False
        ) 

        solva_dir.structure_file = packmol_solvate_wrapper( # generate and point to solvated PDB structure
            polymer_pdb = self.structure_file, 
            solvent_pdb = solvent.structure_file, 
            outdir      = solva_dir.structures, 
            outname     = solva_dir.mol_name, 
            template_path = template_path,
            N           = round(solva_dir.box_volume * solvent.number_density),
            box_dims    = solva_dir.box_vectors,
            precision   = precision
        )

        if solva_dir.has_monomer_info_uncharged: # inject solvent into uncharged monomer file, if present
            solv_mono_data_unchg = solva_dir.monomer_info_uncharged
            solv_mono_data_unchg += solvent.monomer_info 
            solv_mono_data_unchg.to_file(solva_dir.monomer_file_uncharged) # overwrite monomer file with augmented monomer data

        if solva_dir.has_monomer_info_charged: # inject solvent into uncharged monomer file, if present
            solv_mono_data_chg = solva_dir.monomer_info_charged
            solv_mono_data_chg += solvent.monomer_info 
            solv_mono_data_chg.to_file(solva_dir.monomer_file_charged) # overwrite monomer file with augmented monomer data

        solva_dir.solvent = solvent # set this only AFTER solvated files have been created
        solva_dir.to_file() # ensure checkpoint file is created for newly solvated directory
        LOGGER.info('Successfully converged on solvent packing')

    def solvate(self, solvents : Union[Solvent, Iterable[Solvent]], template_path : Path, dest_dir : Optional[Path]=None, exclusion : Optional[float]=None, precision : int=4) -> 'Polymer':
        '''Create a clone of a Polymer and solvate it in a box defined by the polymer's bounding box + an exclusion buffer. Can solvate with one or more solvents'''
        for solvent in general.asiterable(solvents):
            if solvent is not None: # handle the case where a null solvent is passed (technically a valid Solvent for typing reasons)
                self._solvate(solvent=solvent, template_path=template_path, dest_dir=dest_dir, exclusion=exclusion, precision=precision)

# SIMULATION
    def clean_sims(self):
        '''Get rid of any empty (i.e. failed) simulation folders'''
        for sim_dir in self.MD.iterdir():
            if filetree.is_empty(sim_dir):
                sim_dir.rmdir()

    @property
    def completed_sims(self) -> list[Path]:
        '''Return paths to all extant, non-empty simulation subdirectories'''
        self.clean_sims() # discard empty simulation folders
        return [sim_dir for sim_dir in self.MD.iterdir()]
    
    @property
    def chrono_sims(self) -> list[Path]:
        '''Return paths of all extant simulation subdirectories in chronological order of creation'''
        return sorted(self.completed_sims, key=lambda path : self.DATE_FMT.extract_datetime(path.stem))
    
    @property
    def oldest_sim_dir(self) -> Path:
        '''Return the least recent simulation subdir'''
        if not self.chrono_sims:
            raise NoSimulationsFoundError
        return self.chrono_sims[0]

    @property
    def newest_sim_dir(self) -> Path:
        '''Return the most recent simulation subdir'''
        if not self.chrono_sims:
            raise NoSimulationsFoundError
        return self.chrono_sims[-1]
    
    @property
    def simulation_paths(self) -> dict[Path, Path]:
        '''Dict of all extant simulation dirs and their internal path reference files'''
        sim_paths = {}
        for sim_dir in self.completed_sims:
            try:
                sim_paths[sim_dir] = next(sim_dir.glob('*_paths.json'))
            except StopIteration:
                sim_paths[sim_dir] = None
        
        return sim_paths
    
    def make_sim_dir(self, affix : Optional[str]='') -> Path:
        '''Create a new timestamped simulation results directory'''
        sim_name = f'{affix}{"_" if affix else ""}{self.DATE_FMT.timestamp_now()}'
        sim_dir = self.MD/sim_name
        sim_dir.mkdir(exist_ok=False) # will raise FileExistsError in case of overlap
        LOGGER.info(f'Created new Simulation directory "{sim_name}"')

        return sim_dir
    
    def interchange(self, forcefield_path : Path, charge_method : str, periodic : bool=True):
        '''Create an Interchange object for a SMIRNOFF force field using internal structural files'''
        off_topology = self.off_topology_matched() # self.off_topology
        if periodic: # set box vector to allow for periodic simulation (will be non-periodic if polymer box vectors are unset i.e. NoneType)
            off_topology.box_vectors = self.box_vectors.in_units_of(nanometer) 
  
        if self.solvent is None:
            forcefield = ForceField(forcefield_path) #, allow_cosmetic_attributes=True)
        else:
            forcefield = ForceField(forcefield_path, self.solvent.forcefield_file) #, allow_cosmetic_attributes=True)

        self.assign_charges_by_lookup(charge_method) # assign relevant charges prior to returning molecule
        LOGGER.info(f'Creating SMIRNOFF Interchange for "{self.mol_name}" with forcefield "{forcefield_path.stem}"')

        return forcefield.create_interchange(topology=off_topology, charge_from_molecules=[self.offmol]) # package FF, topology, and molecule charges together and ship out
        
    def load_sim_paths_and_params(self, sim_dir : Optional[Path]=None) -> tuple[SimulationPaths, SimulationParameters]:
        '''Takes a path to a simulation directory and returns the associated simulation file paths and parameters
        If no path is provided, will use most recent simulation by default'''
        if sim_dir is None:
            sim_dir = self.newest_sim_dir # use most recent simulation by default

        sim_paths_file = self.simulation_paths[sim_dir] # will raise KeyError if no such simulation exists
        sim_paths = SimulationPaths.from_file(sim_paths_file)
        sim_params = SimulationParameters.from_file(sim_paths.sim_params)

        return sim_paths, sim_params

    def load_traj(self, sim_dir : Optional[Path]=None, **kwargs) -> Trajectory:
        '''Load a trajectory for a simulation directory'''
        if sim_dir is None:
            sim_dir = self.newest_sim_dir # use most recent simulation by default

        sim_paths, sim_params = self.load_sim_paths_and_params(sim_dir)
        return load_traj(sim_paths.trajectory, topo_path=self.structure_file, **kwargs)
    
    def filter_sim_dirs(self, conditions : Union[SimDirFilter, Iterable[SimDirFilter]]) -> dict[Path, tuple[SimulationPaths, SimulationParameters]]:
        '''Returns all simulation directories which meet some binary condition based on their simulation file paths and/or the parameters with which the simulation was ran'''
        conditions = general.asiterable(conditions)
        valid_sim_dirs = {}
        for sim_dir in self.completed_sims:
            sim_paths, sim_params = self.load_sim_paths_and_params(sim_dir)
            if all(condition(sim_paths, sim_params) for condition in conditions):
                valid_sim_dirs[sim_dir] = (sim_paths, sim_params)

        return valid_sim_dirs

    def purge_sims(self, really : bool=False) -> None:
        '''Empties all extant simulation folders - MAKE SURE YOU KNOW WHAT YOU'RE DOING HERE'''
        if not really:
            raise PermissionError('Please confirm that you really want to clear simulation directories (this can\'t be undone!)')
        
        if self.completed_sims: # only attempt removal and logging if simulations are actually present
            filetree.clear_dir(self.MD)
            filetree.clear_dir(self.logs)
            LOGGER.warning(f'Deleted all extant simulations for {self.mol_name}')
