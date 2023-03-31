# Custom Imports
from . import general, filetree
from .solvation import packmol_solvate_wrapper
from .solvents import Solvent
from .extratypes import SubstructSummary
from .charging.types import ResidueChargeMap
from .charging.averaging import write_lib_chgs_from_mono_data
from .charging.application import MolCharger

# Typing and Subclassing
from numpy import number, ndarray
from typing import Any, Callable, ClassVar, Optional
from dataclasses import dataclass, field

# File I/O
from pathlib import Path
from shutil import copyfile
import json, pickle

# Cheminformatics and MD
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from openff.toolkit.topology import Topology, Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

# Units and Quantities
from openmm.unit import angstrom, nanometer


# Polymer representation classes
@dataclass
class PolymerInfo:
    '''For storing useful info about a Polymer Directory'''
    mol_name  : str = field(default=None)
    exclusion : float = field(default=1*nanometer) # distance between molecule bounding box and simulation box size
    solvent   : Solvent = field(default=None) # kept default as NoneType rather than empty string for more intuitive solvent search with unsolvated molecules
    charges   : dict[str, ndarray[float]] = field(default_factory=dict, repr=False)

    ff_file              : Optional[Path] = field(default=None) # .OFFXML
    monomer_file         : Optional[Path] = field(default=None) # .JSON
    monomer_file_chgd    : Optional[Path] = field(default=None) # .JSON
    structure_file       : Optional[Path] = field(default=None) # .PDB
    structure_files_chgd : Optional[dict[str, Path]] = field(default_factory=dict) # dict of .SDF

@dataclass
class PolymerDir:
    '''For representing standard directory structure and requisite information for polymer structures, force fields, and simulations'''
    parent_dir : Path
    mol_name : str

    path            : Path = field(init=False, repr=False)
    checkpoint_path : Path = field(init=False, repr=False)
    info     : PolymerInfo = field(init=False, default_factory=PolymerInfo)
    charge_method    : str = field(init=False, default=None)

    subdirs : list[Path] = field(default_factory=list, repr=False)
    _SUBDIRS : ClassVar[tuple[str, ...]] = ( # directories with these names will be present in all polymer directories by standard
        'structures',
        'monomers',
        'SDF',
        'FF',
        'MD',
        'checkpoint',
        'logs'
    )

    _off_topology : Topology = field(default=None, init=False) # for caching purposes
    _offmol       : Molecule = field(default=None, init=False)

# CONSTRUCTION 
    def __post_init__(self):
        '''Initialize core directory and file paths'''
        self.path = self.parent_dir/self.mol_name
        self.info.mol_name = self.mol_name # slightly redundant but more modular

        for dir_name in PolymerDir._SUBDIRS:
            subdir = self.path/dir_name
            setattr(self, dir_name, subdir)
            self.subdirs.append(subdir)

        self.checkpoint_path = self.checkpoint/f'{self.mol_name}_checkpoint.pkl'
        self.build_tree()

    def build_tree(self) -> None:
        '''Build the main directory and tree '''
        self.path.mkdir(exist_ok=True)
        for subdir in self.subdirs:
            subdir.mkdir(exist_ok=True)

        self.checkpoint_path.touch() # must be done AFTER subdirectory creation, since the checkpoint file resides in the "checkpoint" subdirectory

    def empty(self) -> None:
        '''
        Undoes build_tree - intended to "reset" a directory
        NOTE : will break most functionality if build_tree() is not subsequently called
        '''
        filetree.clear_dir(self.path)

# CHECKPOINT FILE I/O
    def to_file(self) -> None:
        '''Save directory object to disc - for checkpointing and non-volatility'''
        with self.checkpoint_path.open('wb') as checkpoint_file:
            pickle.dump(self, checkpoint_file)

    @classmethod
    def from_file(cls, checkpoint_path : Path) -> 'PolymerDir':
        '''Load a saved directory tree object from disc'''
        assert(checkpoint_path.suffix == '.pkl')
        with checkpoint_path.open('rb') as checkpoint_file:
            return pickle.load(checkpoint_file)
        
    def update_checkpoint(funct : Callable) -> Callable[[Any], Optional[Any]]: # NOTE : this deliberately doesn't have a "self" arg!
        '''Decorator for updating the on-disc checkpoint file after a function updates a PolymerDir attribute'''
        def update_fn(self, *args, **kwargs) -> Optional[Any]:
            ret_val = funct(self, *args, **kwargs) # need temporary value so update call can be made before returning
            self.to_file()
            return ret_val
        return update_fn
        
# STRUCTURE FILE PROPERTIES
    @property
    def monomer_file_ranked(self) -> Path:
        '''Choose monomer file from those available according to a ranked priority - returns Nonetype if no files are specified'''
        MONO_PRIORITY_ORDER = ('monomer_file_chgd', 'monomer_file') # NOTE : order here isn't arbitrary, establishes FIFO priority for monomer files - must match named PolymerInfo attributes

        for mono_file_type in MONO_PRIORITY_ORDER:
            if (possible_monofile := getattr(self.info, mono_file_type)) is not None:
                return possible_monofile
        else:
            return None

    @property
    def monomer_data(self) -> dict[str, Any]:
        '''Load monomer information from file'''
        if self.info.monomer_file is None:
            raise FileExistsError(f'No monomer file exists for {self.mol_name}')

        with self.info.monomer_file.open('r') as json_file: 
            return json.load(json_file)
    
    @property
    def monomer_data_charged(self) -> dict[str, Any]:
        '''Load monomer information with charges from file'''
        if self.info.monomer_file is None:
            raise FileExistsError(f'No monomer file with charged exists for {self.mol_name}')

        with self.info.monomer_file_chgd.open('r') as json_file: 
            return json.load(json_file)

    @property
    def has_monomer_data(self) -> bool:
        return (self.monomer_file_ranked is not None)

    @property
    def has_structure_data(self) -> bool:
        return (self.info.structure_file is not None)
    
# RDKit PROPERTIES 
    @property
    def rdmol_topology(self) -> RDMol:
        '''Load an RDKit Molecule object directly from structure file'''
        return Chem.MolFromPDBFile(str(self.info.structure_file), removeHs=False)
    
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
        return self.mol_bbox + 2*self.info.exclusion # 2x accounts for the fact that exclusion must occur on either side of the tight bounding box
    
# OpenFF / OpenMM PROPERTIES
    def off_topology_matched(self, strict : bool=True, verbose : bool=False, chgd_monomers : bool=False, topo_only : bool=True) -> tuple[Topology, Optional[list[SubstructSummary]], Optional[bool]]:
        '''Performs a monomer substructure match and returns the resulting OpenFF Topology'''
        if chgd_monomers:
            assert(self.info.monomer_file_chgd.exists())
            off_topology, substructs, error_state = Topology.from_pdb_and_monomer_info(str(self.info.structure_file), self.info.monomer_file_chgd, strict=strict, verbose=verbose)
        else:
            assert(self.info.monomer_file.exists())
            off_topology, substructs, error_state = Topology.from_pdb_and_monomer_info(str(self.info.structure_file), self.info.monomer_file, strict=strict, verbose=verbose)

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
        if self._off_topology is None:
            self._off_topology = self.off_topology_matched() # cache topology match
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._off_topology
    
    @property 
    def largest_offmol(self):
        if self._offmol is None:
            self._offmol = self.largest_offmol_matched()
            self.to_file() # not using "update_checkpoint" decorator here since attr write happens only once, and to avoid interaction with "property" 
        return self._offmol
    offmol = largest_offmol # alias for simplicity

# CHARGING
    @update_checkpoint
    def register_charges(self, charge_method : str, charges : ndarray[float]) -> None:
        '''For inserting/overwriting available charge sets in local session and on file'''
        assert(general.hasunits(charges)) # ensure charges have units (can't assign to Molecules if not)
        self.info.charges[charge_method] = charges

    @update_checkpoint
    def unregister_charges(self, charge_method) -> None:
        '''For removing available charge sets in local session and on file'''
        self.info.charges.pop(charge_method, None) # TOSELF : can get rid of None default arg if KeyError is istaed desired

    @update_checkpoint
    def assign_charges_by_lookup(self, charge_method : str) -> None:
        '''Choose which registered charges should be applied to the cached Molecule by key lookup'''
        self.offmol.partial_charges = self.info.charges[charge_method] # NOTE : explicitly using property version of offmol (i.e. not _offmol) in case Molecule isn't yet instantiated
        self.charge_method = charge_method 
    
    def assign_charges(self, charge_method : str, charges : ndarray[float]) -> None:
        '''Wrap registration and assignment for convenience'''
        self.register_charges(charge_method, charges) # doesn't need "update_checkpoint" flag, as all internal methods already do so
        self.assign_charges_by_lookup(charge_method)

    @update_checkpoint
    def charge_and_save_molecule(self, charger : MolCharger, *topo_args, **topo_kwargs) -> tuple[Molecule, Path]:
        '''Generates and registers 1) an .SDF file for the charged molecule and 2) an charge entry for just the Molecule's charges'''
        charge_method = charger.TAG
        unchgd_mol = self.offmol_matched(*topo_args, **topo_kwargs) # load a fresh, uncharged molecule from graph match (important to avoid charge contamination)
        chgd_mol = charger.charge_molecule(unchgd_mol)
        self.register_charges(charge_method, chgd_mol.partial_charges)

        sdf_path = self.SDF/f'{self.mol_name}_charged_{charge_method}.sdf'
        chgd_mol.properties['metadata'] = [atom.metadata for atom in chgd_mol.atoms] # need to store metadata as separate property, since SDF does not preserved metadata atomwise
        chgd_mol.to_file(str(sdf_path), file_format='SDF')
        self.info.structure_files_chgd[charge_method] = sdf_path

        return chgd_mol, sdf_path

# FILE POPULATION AND MANAGEMENT
    @update_checkpoint
    def populate_mol_files(self, source_dir : Path) -> None:
        '''
        Populates a PolymerDir with the relevant structural and monomer files from a shared source ("data dump") folder
        Assumes that all structure and monomer files will have the same name as the PolymerDir in question
        '''
        pdb_path = source_dir/f'{self.mol_name}.pdb'
        new_pdb_path = self.structures/f'{self.mol_name}.pdb'
        copyfile(pdb_path, new_pdb_path)
        self.info.structure_file = new_pdb_path

        monomer_path = source_dir/f'{self.mol_name}.json'
        if monomer_path.exists():
            new_monomer_path = self.monomers/monomer_path.name
            copyfile(monomer_path, new_monomer_path)
            self.info.monomer_file = new_monomer_path
    
    @update_checkpoint
    def create_charged_monomer_file(self, residue_charges : ResidueChargeMap):
        '''Create a new copy of the current monomer file with residue-wise mapping of substructure id to charges'''
        assert(self.info.monomer_file is not None)

        if self.info.solvent is not None and self.info.solvent.charges is not None: # ensure solvent "monomer" charge entries are also recorded
            residue_charges = {**residue_charges, **self.info.solvent.monomer_json_data['charges']} 
            
        chgd_monomer_data = {**self.monomer_data} # load uncharged monomer_data and create copy
        chgd_monomer_data['charges'] = residue_charges

        chgd_mono_path = self.info.monomer_file.with_name(f'{self.mol_name}_charged.json')
        with chgd_mono_path.open('w') as chgd_json:
            json.dump(chgd_monomer_data, chgd_json, indent=4)

        self.info.monomer_file_chgd = chgd_mono_path # record path to new json file

    @update_checkpoint
    def solvate(self, template_path : Path, solvent : Solvent, exclusion : float=None, precision : int=4) ->  'PolymerDir':
        '''Applies packmol solvation routine to an extant PolymerDir'''
        assert(self.has_structure_data) # TODO : clean these checks up eventually
        assert(solvent.structure_file is not None)

        if exclusion is None:
            exclusion = self.info.exclusion # default to same exclusion as parent

        solvated_name = f'{self.mol_name}_solv_{solvent.name}'
        solvated_dir = PolymerDir(parent_dir=self.parent_dir, mol_name=solvated_name)

        solvated_dir.info.exclusion = exclusion
        box_vectors = self.mol_bbox + 2*exclusion # can't use either dirs "box_vectors" property directly, since the solvated dir has no structure file yet and the old dir may have different exclusion
        V_box = general.product(box_vectors)

        solvated_dir.info.structure_file = packmol_solvate_wrapper( # generate and point to solvated PDB structure
            polymer_pdb = self.info.structure_file, 
            solvent_pdb = solvent.structure_file, 
            outdir      = solvated_dir.structures, 
            outname     = solvated_name, 
            template_path = template_path,
            N           = round(V_box * solvent.number_density),
            box_dims    = box_vectors,
            precision   = precision
        )

        if self.has_monomer_data:
            solvated_mono_data = {**self.monomer_data}           # note that this is explicitly UNCHARGED monomer data
            for field, values in solvated_mono_data.items():     # this merge strategy ensures solvent data does not overwrite or append extraneous data
                values.update(solvent.monomer_json_data[field])  # specifically, charges will not be written to an uncharged json (which would screw up graph match and load)

            solva_mono_path = solvated_dir.monomers/f'{solvated_name}.json'
            with solva_mono_path.open('w') as solv_mono_file:
                json.dump(solvated_mono_data, solv_mono_file, indent=4)
            solvated_dir.info.monomer_file = solva_mono_path

        solvated_dir.info.solvent = solvent # set this only AFTER solvated files have been created
        solvated_dir.to_file() # ensure checkpoint file is created for newly solvated directory
        return solvated_dir
    
    @update_checkpoint
    def create_FF_file(self, xml_src : Path) -> ForceField:
        '''Generate an OFF force field with molecule-specific (and solvent specific, if applicable) Library Charges appended'''
        assert(self.info.monomer_file_chgd is not None)

        ff_path = self.FF/f'{self.mol_name}.offxml' # path to output library charges to
        forcefield, lib_chgs = write_lib_chgs_from_mono_data(self.monomer_data_charged, xml_src, output_path=ff_path)

        if self.info.solvent is not None:
            forcefield = ForceField(ff_path, self.info.solvent.forcefield_file, allow_cosmetic_attributes=True) # use both the polymer-specific xml and the solvent FF xml to make hybrid forcefield
            forcefield.to_file(ff_path)

        self.info.ff_file = ff_path # ensure change is reflected in directory info
        return forcefield

# SIMULATION
    @property
    def completed_sims(self) -> list[Path]:
        '''Return paths to all non-empty simulation subdirectories'''
        return [sim_dir for sim_dir in self.MD.iterdir() if not filetree.is_empty(sim_dir)]

    def make_res_dir(self, affix : Optional[str]='') -> Path:
        '''Create a new timestamped simulation results directory'''
        res_name = f'{affix}{"_" if affix else ""}{general.timestamp_now()}'
        res_dir = self.MD/res_name
        res_dir.mkdir(exist_ok=False) # will raise FileExistsError in case of overlap
        return res_dir
    
    def purge_sims(self, really : bool=False) -> None:
        '''Empties all extant simulation folders - MAKE SURE YOU KNOW WHAT YOU'RE DOING HERE'''
        if not really:
            raise PermissionError('Please confirm that you really want to clear simulation directories (this can\'t be undone!)')
        filetree.clear_dir(self.MD)
        filetree.clear_dir(self.logs)


class PolymerDirManager:
    '''Class for organizing, loading, and manipulating collections of PolymerDir objects'''
    def __init__(self, collection_dir : Path):
        self.collection_dir : Path = collection_dir
        self.log_dir : Path = self.collection_dir/'Logs'
        self.mol_dirs_list : list[PolymerDir] = []

        self.collection_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.update_mol_dirs() # populate currently extant dirs

    def update_mol_dirs(self, return_missing : bool=False) -> Optional[list[Path]]:
        '''
        Load existing polymer directories from target directory into memory
        Return a list of the checkpoint patfhs which contain faulty/no data
        '''
        found_mol_dirs, missing_chk_data = [], []
        for checkpoint_file in self.collection_dir.glob('**/*_checkpoint.pkl'):
            try:
                found_mol_dirs.append( PolymerDir.from_file(checkpoint_file) )
            except EOFError:
                missing_chk_data.append(checkpoint_file)
                # print(f'Missing checkpoint file info in {mol_dir}') # TODO : find better way to log this
        self.mol_dirs_list = found_mol_dirs # avoiding direct append ensures no double-counting if run multiple times (starts with empty list each time)

        if return_missing:
            return missing_chk_data
        
    def auto_update(funct : Callable) -> Callable[[Any], Optional[Any]]: # NOTE : this deliberately doesn't have a "self" arg!
        '''Decorator for updating the internal list of PolymerDirs after a function'''
        def update_fn(self, *args, **kwargs) -> Optional[Any]:
            ret_val = funct(self, *args, **kwargs) # need temporary value so update call can be made before returning
            self.update_mol_dirs(return_missing=False)
            return ret_val
        return update_fn
    
    @property
    def mol_dirs(self) -> dict[str, PolymerDir]:
        '''
        Key present PolymerDirs by their molecule name (convenient for applications where frequent name lookup/str manipulation is required)
        Written as a property in order to generate unique copies of the dict when associating to another variable (prevents shared overwrites)
        '''
        return {mol_dir.mol_name : mol_dir for mol_dir in self.mol_dirs_list}
    
    @property
    def all_completed_sims(self) -> dict[str, list[Path]]: 
        '''Get a dict of currently populated simulation folders itermized by molecule name'''
        return {
            mol_name : mol_dir.completed_sims
                for mol_name, mol_dir in self.mol_dirs.items()
                    if mol_dir.completed_sims # don't report directories without simulations
        }

    @auto_update
    def populate_mol_dirs(self, source_dir : Path) -> None:
        '''Generate PolymerDirs via files extracted from a lumped PDB/JSON directory'''
        for pdb_path in source_dir.glob('**/*.pdb'):
            mol_name = pdb_path.stem
            parent_dir = self.collection_dir/mol_name
            parent_dir.mkdir(exist_ok=True)

            mol_dir = PolymerDir(parent_dir, mol_name)
            mol_dir.populate_mol_files(source_dir=source_dir)

    def purge_logs(self, really : bool=False) -> None: # NOTE : deliberately undecorated, as no PolymerDir reload is required
        if not really:
            raise PermissionError('Please confirm that you really want to clear all directories (this can\'t be undone!)')
        filetree.clear_dir(self.log_dir)

    @auto_update
    def purge_dirs(self, really : bool=False) -> None:
        if not really:
            raise PermissionError('Please confirm that you really want to clear all directories (this can\'t be undone!)')
        filetree.clear_dir(self.collection_dir)
        self.log_dir.mkdir(exist_ok=True) # remake Log directory
    
    @auto_update
    def purge_sims(self, really : bool=False) -> None:
        if not really:
            raise PermissionError('Please confirm that you really want to clear all directories (this can\'t be undone!)')
        for mol_dir in self.mol_dirs_list:
            mol_dir.purge_sims(really=really)
        self.purge_logs(really=really)