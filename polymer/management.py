'''Utilities and classes for managing ordered hierarchical collections of on-disc Polymer objects'''

# Logging
import logging
from logging import Logger
LOGGER = logging.getLogger(__name__)

# Custom imports
from .. import LOGGERS_MASTER
from .. import extratypes, filetree
from ..logutils import ProcessLogHandler

from .representation import Polymer
from . import filters

# Typing and Subclassing
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, TypeAlias, Union

from ..solvation.solvent import Solvent
PolymerFunction : TypeAlias = Callable[[Polymer, logging.Logger, Any], None]

# File I/O
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction, Namespace


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
                LOGGER.error(f'Missing checkpoint file data in {checkpoint_file}') 
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
    
    def logging_wrapper(self, proc_name : Optional[str]=None, filters : Optional[Union[filters.MolFilter, Iterable[filters.MolFilter]]]=None, loggers : Union[Logger, list[Logger]]=LOGGERS_MASTER) -> Callable[[Callable], Callable]: # NOTE : this is deliberately NOT a staticmethod
        '''Decorator for wrapping an action over a polymer into a logged loop over all present polymers
        Can optionally specify a set of filters to only apply the action to a subset of the polymers present
        Logs generated at both the individual polymer level and the global Manager level (messages get passed upwards)'''
        if filters is None:
            filters = [filters.identity] # no filtering if not explicitly specified
        sample_dirs = self.filtered_by(filters)

        # TODO : guarantee that the local LOGGER is always present, regardless of the logger(s) passed

        def logging_decorator(funct : PolymerFunction) -> Callable[..., Any]:
            def wrapper(*args, **kwargs) -> None:
                with ProcessLogHandler(filedir=self.log_dir, loggers=loggers, proc_name=proc_name, timestamp=True) as msf_handler:
                    for i, (mol_name, polymer) in enumerate(sample_dirs.items()):
                        LOGGER.info(f'Current molecule: "{mol_name}" ({i + 1}/{len(sample_dirs)})') # +1 converts to more human-readable 1-index for step count
                        with msf_handler.subhandler(filedir=polymer.logs, loggers=loggers, proc_name=f'{proc_name} for {mol_name}', timestamp=True) as subhandler: # also log actions to individual Polymers
                            funct(polymer, subhandler.personal_logger, *args, **kwargs)
            return wrapper
        return logging_decorator
    
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

    def filtered_by(self, filters : Union[filters.MolFilter, Iterable[filters.MolFilter]]) -> dict[str, Polymer]:
        '''Return name-keyed dict of all Polymers in collection which meet all of a set of filtering conditions'''
        filters = extratypes.asiterable(filters)
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
            LOGGER.info('') # leave a line break between polymer entries to give some breathing room

    @refresh_listed_dirs # TODO : reimplement with logging wrapper method
    def solvate_collection(self, solvents : Iterable[Solvent], template_path : Path, exclusion : float=None) -> None:
        '''Make solvated versions of all listed Polymers, according to a collection of solvents'''      
        for polymer in self.polymers_list:
            polymer.solvate(solvents=solvents, template_path=template_path, exclusion=exclusion)
            LOGGER.info('') # leave a line break between polymer entries to give some breathing room

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

@dataclass
class NameFilterBuffer:
    '''Class for eliminating boilerplate when filtering Polymer collection-based scripts'''
    molecules : Optional[Iterable[str]] = None # base names of molecules to select for
    solvent   : Optional[bool] = None # whether to express a preference for having solvent
    charges   : Optional[bool] = None # whether to express a preference for having charges
        
    @property
    def filters(self) -> list[filters.MolFilter]:
        '''Generates list of relevant filters based on current setting'''
        filters = []
        if self.molecules: # NOTE : not explicitly checking for NoneType, as empty iterables should also be skipped
            desired_mol = filters.filter_factory_by_attr('base_mol_name', lambda name : name in self.molecules)
            filters.append(desired_mol)

        if self.charges is not None:
            filters.append(filters.is_charged if self.solvent else filters.is_uncharged)

        if self.solvent is not None:
            filters.append(filters.is_solvated if self.solvent else filters.is_unsolvated)

        return filters

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('--molecules', help='List of molecule types (by name) to select for', nargs='+', action='store')
        parser.add_argument('--solvent'  , help='Set which solvation type to filter for (options are "solv", "unsolv", or "all", defaults to "all")', action=BooleanOptionalAction)
        parser.add_argument('--charges'  , help='Set which charging status to filter for (options are "chg", "unchg", or "all", defaults to "all")' , action=BooleanOptionalAction)

    @classmethod
    def from_argparse(cls, args : Namespace) -> 'NameFilterBuffer':
        '''Initialize from an argparse Namespace'''
        return cls(
            molecules=args.molecules,
            solvent=args.solvent,
            charges=args.charges
        )

    def valid_names(self, poly_mgr : PolymerManager) -> list[str]:
        '''Returns names of Polymers in a collection which fit all initialized criteria'''
        return [mol_name for mol_name in poly_mgr.filtered_by(self.filters)]

    def create_filter_for(self, poly_mgr : PolymerManager) -> filters.MolFilter:
        '''Generates a filter which '''
        return filters.filter_factory_by_attr('mol_name', condition=lambda name : name in self.valid_names(poly_mgr))