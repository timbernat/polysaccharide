# Custom Imports
from . import general, filetree
from .solvation import packmol_solvate_wrapper
from .solvents import Solvent

# Typing and Subclassing
from typing import ClassVar, Optional
from dataclasses import dataclass, field

# File I/O
from pathlib import Path
import json, pickle
from shutil import copyfile

# Cheminformatics
from rdkit import Chem

# Units and Quantities
from openmm.unit import angstrom, nanometer


# Polymer representation classes
@dataclass
class PolymerInfo:
    '''For storing useful info about a Polymer Directory'''
    mol_name          : str = field(default=None)
    exclusion         : float = field(default=1*nanometer) # distance between molecule bounding box and simulation box size
    solvent           : Solvent = field(default=None) # kept default as NoneType rather than '' for more intuitive solvent search with unsolvated molecules

    structure_file    : Optional[Path] = field(default=None)
    monomer_file      : Optional[Path] = field(default=None)
    monomer_file_chgd : Optional[Path] = field(default=None)
    pickle_file       : Optional[Path] = field(default=None) # TOSELF : this stores pickled Molecule() objects, NOT pickled PolymerDir() checkpoints
    ff_file           : Optional[Path] = field(default=None)

@dataclass
class PolymerDir:
    '''For representing standard directory structure and requisite information for polymer structures, force fields, and simulations'''
    parent_dir : Path
    mol_name : str

    path : Path = field(init=False)
    checkpoint_path : Path = field(init=False)
    info : PolymerInfo = field(init=False)

    subdirs : list[Path] = field(default_factory=list)

    _SUBDIRS : ClassVar[tuple[str, ...]] = ( # directories with these names will be present in all polymer directories by standard
        'structures',
        'monomers',
        'pkl',
        'FF',
        'MD',
        'checkpoint',
        'logs'
    )

# CONSTRUCTION 
    def __post_init__(self):
        '''Initialize core directory and file paths'''
        self.path = self.parent_dir/self.mol_name
        self.info = PolymerInfo() # will keep file updated as object is updated
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

# FILE I/O
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
        
# PROPERTIES AND ATTRIBUTE CALCULATIONS
    @property
    def monomer_file_ranked(self):
        '''Choose monomer file from those available according to a ranked priority - returns Nonetype if no files are specified'''
        MONO_PRIORITY_ORDER = ('monomer_file_chgd', 'monomer_file') # NOTE : order here isn't arbitrary, establishes FIFO priority for monomer files - must match named PolymerInfo attributes

        for mono_file_type in MONO_PRIORITY_ORDER:
            if (possible_monofile := getattr(self.info, mono_file_type)) is not None:
                return possible_monofile
        else:
            return None

    @property
    def has_monomer_data(self) -> bool:
        return (self.monomer_file_ranked is not None)

    @property
    def has_structure_data(self) -> bool:
        return (self.info.structure_file is not None)
    
    @property
    def rdmol(self):
        '''Load an RDKit Molecule object directly from structure file'''
        return Chem.MolFromPDBFile(str(self.info.structure_file), removeHs=False)
    
    @property
    def largest_mol(self):
        '''
        Return the largest sub-molecule in the structure file Topology
        Intended to differentiate target molecules from solvent
        '''
        return max(Chem.rdmolops.GetMolFrags(self.rdmol, asMols=True), key=lambda mol : mol.GetNumAtoms())
    mol = largest_mol # alias for simplicity

    @property
    def n_atoms(self):
        '''Number of atoms in the main polymer chain (excluding solvent)'''
        return self.mol.GetNumAtoms()

    @property
    def mol_bbox(self):
        '''Return the bounding box size (in angstroms) of the molecule represented'''
        return self.mol.GetConformer().GetPositions().ptp(axis=0) * angstrom
    
    @property
    def box_vectors(self):
        '''Dimensions fo the periodic simulation box'''
        return self.mol_bbox + self.info.exclusion
    
# SIMULATION
    def make_res_dir(self, affix : Optional[str]='') -> Path:
        '''Create a new timestamped simulation results directory'''
        res_name = f'{affix}{"_" if affix else ""}{general.timestamp_now()}'
        res_dir = self.MD/res_name
        res_dir.mkdir(exist_ok=False) # will raise FileExistsError in case of overlap
        return res_dir
    
    def _purge_sims(self) -> None:
        '''Empties all extant simulation folders - MAKE SURE YOU KNOW WHAT YOU'RE DOING HERE'''
        filetree.clear_dir(self.MD)
    
# FILE POPULATION AND MANAGEMENT
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

        self.to_file() # ensure disk copy is updated appropriately
    
    def solvate(self, template_path : Path, solvent : Solvent, exclusion : float=None, precision : int=4) ->  'PolymerDir':
        '''Applies packmol solvation routine to an extant PolymerDir'''
        assert(self.has_structure_data) # TODO : clean these check up eventually
        assert(solvent.structure_file is not None)

        if exclusion is None:
            exclusion = self.info.exclusion # default to same exclusion as parent

        outname = f'{self.mol_name}_solv_{solvent.name}'
        solvated_dir = PolymerDir(parent_dir=self.parent_dir, mol_name=outname)
        solvated_dir.info.exclusion = exclusion

        box_vectors = self.mol_bbox + exclusion # can't use either dirs "box_vectors" property directly, since the solvated dir has no structure file yet and the old dir may have different exclusion
        V_box = general.product(box_vectors)

        solvated_dir.info.structure_file = packmol_solvate_wrapper( # generate and point to solvated PDB structure
            polymer_pdb=self.info.structure_file, 
            solvent_pdb=solvent.structure_file, 
            outdir=solvated_dir.structures, 
            outname=outname, 
            template_path=template_path,
            N=round(V_box * solvent.number_density),
            box_dims=box_vectors,
            precision=precision
        )

        if self.has_monomer_data:
            mono_file = self.monomer_file_ranked
            with mono_file.open('r') as mono_src:
                mono_data = json.load(mono_src)
            
            for field, values in mono_data.items(): # this merge strategy ensures solvent data does not overwrite or append extraneous data
                values.update(solvent.monomer_json_data[field])  # specifically, charges will not be written to an uncharged json (which would screw up graph match and load)

            new_mono_path = solvated_dir.monomers/f'{outname}.json'
            new_mono_path.touch()
            with new_mono_path.open('w') as mono_out:
                json.dump(mono_data, mono_out, indent=4)
            solvated_dir.info.monomer_file = new_mono_path

        solvated_dir.info.solvent = solvent # set this only AFTER solvated files have been created
        solvated_dir.to_file() # ensure data is written to disk
        
        return solvated_dir