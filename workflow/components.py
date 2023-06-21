'''Individual units representing tasks which can be strung together to define a Process'''

# Logging
import logging

# Typing and Subclassing
from typing import Any, Callable, ClassVar, Iterable, Optional, Union
from abc import ABC, abstractmethod, abstractproperty

# Generic imports
import datetime
from pathlib import Path
from shutil import copyfile
from time import sleep
from abc import abstractstaticmethod

import re
import sys, subprocess
from argparse import ArgumentParser, Namespace

# Resource imports
import importlib_resources as impres
from polysaccharide import resources

avail_chg_templates = ', '.join(
    path.name
        for path in resources.AVAIL_RESOURCES['chg_templates']
)

avail_sim_templates = ', '.join(
    path.name
        for path in resources.AVAIL_RESOURCES['sim_templates']
)

# Polymer Imports
from polysaccharide.general import hasunits, asiterable
from polysaccharide.filetree import default_suffix

from polysaccharide.solvation.solvent import Solvent
from polysaccharide.solvation import solvents as all_solvents

from polysaccharide.charging.application import ChargingParameters, CHARGER_REGISTRY
from polysaccharide.charging.averaging import get_averaged_residue_charges, AveragingCharger

from polysaccharide.simulation.records import SimulationParameters
from polysaccharide.simulation.ensemble import EnsembleSimulationFactory
from polysaccharide.simulation.execution import run_simulation

from polysaccharide.polymer.representation import Polymer
from polysaccharide.polymer.management import PolymerManager, PolymerFunction, MolFilterBuffer
from polysaccharide.polymer.filtering import MolFilter, has_sims, has_monomers_chgd, is_base
from polysaccharide.polymer.filtering import SimDirFilter, has_binary_traj

from polysaccharide.polymer.monomer import estimate_max_DOP, estimate_chain_len
from polysaccharide.polymer.building import build_linear_polymer
from polysaccharide.polymer.exceptions import ExcessiveChainLengthError

from polysaccharide.analysis import trajectory

# Molecular Dynamics
from openmm.unit import nanosecond # time
from openmm.unit import nanometer, angstrom # length


# Base class
class WorkflowComponent(ABC):
    '''Base class for assembling serial and parallel set of actions over Polymers'''
    @abstractproperty
    @classmethod
    def desc(self) -> str:
        '''Brief description to accompany component'''
        ...
    
    @abstractproperty
    @classmethod
    def name(self) -> str:
        '''Brief name to label component'''
        ...

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, desc={self.desc})'

    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abstractstaticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        ...

    @abstractmethod
    def make_polymer_fn(self) -> PolymerFunction:
        ...

    @classmethod
    def process_argparse_args(self, args : Namespace) -> dict[Any, Any]:
        '''For postprocessing argparse str keywords into objects prior to attempting initialization
        If not overridden in concrete child classes, will attempt argparse initialize with unmodified CLI inputs'''
        return vars(args) # return args dict verbatim by default

    @classmethod
    def from_argparse(cls, args : Namespace) -> 'WorkflowComponent':
        '''Initialize from an argparse Namespace'''
        return cls(**cls.process_argparse_args(args))

    def assert_filter_prefs(self, molbuf : MolFilterBuffer) -> list[MolFilter]:
        '''Assert any additional preferences for filters beyond the default molecule filters'''
        return molbuf.filters # default to base filters

    @classmethod
    @property
    def registry(cls) -> dict[str, 'WorkflowComponent']:
        '''Name-indexed dict of all inherited Component implementations'''
        return {
            subcomp.name : subcomp
                for subcomp in cls.__subclasses__()
        }


# Concrete child class implementations
class DummyCalculation(WorkflowComponent):
    desc = 'Computes RDF and property time series data and saving to csvs for plotting and analysis'
    name = 'dummy'

    def __init__(self, wait_time : int, **kwargs):
        '''Initialize wait time to simulate non-trivial task'''
        self.wait_time = wait_time

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-w', '--wait_time', help='Number of seconds for dummy task to wait', action='store', type=int)

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Dummy function for testing script dispatch'''
            poly_logger.info(f'Performing fake calculation for {polymer.mol_name} for {self.wait_time} seconds')
            sleep(self.wait_time)

        return polymer_fn

class CollectionPopulation(WorkflowComponent):
    desc = 'Initialize a collection of Polymers from matching sets of PDB structures and accompanying monomer information'
    name = 'populate'

    def __init__(self, pdb_struct_dir : Path, monomer_dir : Optional[Path], **kwargs):
        self.pdb_struct_dir = pdb_struct_dir
        self.monomer_dir = monomer_dir # if (monomer_dir is not None) else pdb_struct_dir

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        parser.add_argument('-pdb', '--pdb_struct_dir', help='The path of the directory in containing the target pdb structure files', type=Path, required=True)
        parser.add_argument('-mono', '--monomer_dir', help='The path of the directory in containing the accompanying monomer info JSON files', type=Path)

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Source structure and monomer files from directories'''
            polymer.populate_mol_files(self.pdb_struct_dir, monomer_dir=self.monomer_dir)
        
        return polymer_fn
    
class Solvate(WorkflowComponent):
    desc = 'Solvate molecules in sets of 1 or more desired solvents'
    name = 'solvate'

    def __init__(self, solvents : Union[Solvent, Iterable[Solvent]], template_path : Path, exclusion : float=None, **kwargs):
        if not solvents:
            raise ValueError('Must specify at least 1 solvent')

        if not hasunits(exclusion):
            exclusion *= nanometer  # assign units

        self.solvents = solvents
        self.template_path = template_path
        self.exclusion = exclusion

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-s', '--solvents'         , help='Names of all solvent molecule to solvate the target systems in' , action='store', nargs='+', default=['WATER_TIP3P'])
        parser.add_argument('-pt', '--packmol_template', help='Name of the packmol input template file to use for solvation', action='store', default='solv_polymer_template_box.inp')
        parser.add_argument('-dir', '--directory'      , help='Path of the folder in which the chosen packmol solvation input file resides', type=Path, default=impres.files(resources.inp_templates))
        parser.add_argument('-e', '--exclusion'        , help='Distance (in nm) between the bounding box of the molecule and the simiulation / solvation box', action='store', type=float, default=1.0)

    @classmethod
    def process_argparse_args(self, args: Namespace) -> dict[Any, Any]:
        '''Ensure solvents are initialized as Solvent objects'''
        return {
            'solvents' : [
                getattr(all_solvents, solvent_name)
                    for solvent_name in args.solvents
            ],
            'template_path' : default_suffix(args.directory / args.packmol_template, suffix='inp'),
            'exclusion' : args.exclusion
        }

    def assert_filter_prefs(self, molbuf : MolFilterBuffer) -> list[MolFilter]:
        '''Assert any additional preferences for filters beyond the default molecule filters'''
        molbuf.solvent = False # force preference for only unsolvated molecules (don't want to attempt solvation twice)
        return molbuf.filters

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Fill a box around a polymer with solvent'''
            polymer.solvate(self.solvents, template_path=self.template_path, exclusion=self.exclusion)
 
        return polymer_fn
    
class BuildReducedStructures(WorkflowComponent):
    desc = 'Creates directory of reduced-chain-length structure PDBs and monomer JSONs from a collection of linear Polymers'
    name = 'redux'

    def __init__(self, struct_output : Path, mono_output : Path, DOP : Optional[int]=None, max_chain_len : Optional[int]=None, chain_len_limit : int=300, flip_term_labels : Optional[Iterable]=None, **kwargs):
        '''Initialize sizes of new chains to build, along with locations to output structure files to'''
        self.struct_output = struct_output  
        self.mono_output = mono_output 
        self.struct_output.mkdir(exist_ok=True, parents=True)
        self.mono_output.mkdir(  exist_ok=True, parents=True)

        if not (max_chain_len or DOP):
            raise ValueError('Must provide EITHER a maximum chain length OR a degree of polymerization (provided neither)')

        if max_chain_len and DOP:
            raise ValueError('Must provide EITHER a maximum chain length OR a degree of polymerization (provided both)')
        
        self.max_chain_len = max_chain_len  
        self.DOP = DOP  
        self.chain_len_limit = chain_len_limit  
        self.flip_term_labels = flip_term_labels  

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-pdb', '--struct_output'  , help='The name of the directory to output generated PDB structure to', type=Path)
        parser.add_argument('-mono' , '--mono_output'  , help='The name of the directory to output generated JSON monomer files to', type=Path)
        parser.add_argument('-N', '--max_chain_len'    , help='Maximum number of atoms in any of the reduced chain generated. If this is specified, CANNOT specify DOP', type=int)
        parser.add_argument('-D', '--DOP'              , help='The number of monomer units to include in the generated reductions.  If this is specified, CANNOT specify max_chain_len', type=int)
        parser.add_argument('-lim', '--chain_len_limit', help='The maximum allowable size for a chain to be built to; any chains attempted to be built larger than this limit will raise an error', type=int, default=300)
        parser.add_argument('-f', '--flip_term_labels' , help='Names of the chains on which to reverse the order of head/tail terminal group labels (only works for linear homopolymers!)', action='store', nargs='+', default=tuple())

    def assert_filter_prefs(self, molbuf : MolFilterBuffer) -> list[MolFilter]:
        '''Assert any additional preferences for filters beyond the default molecule filters'''
        molbuf.solvent = False # force preference for unsolvated molecules - makes logic for monomer selection cleaner (don;t need to worry about wrong number of monomers due to solvent)
        return molbuf.filters 

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Builds new PDB structures of the desired size from the monomers of an existing Polymer'''

            monomer_smarts = polymer.monomer_info.monomers # create copy to avoid popping from original
            if self.DOP: # NOTE : this only works as intended because of the exclusivity check during arg processing
                max_chain_len = estimate_chain_len(monomer_smarts, DOP)
                DOP = self.DOP

            if self.max_chain_len:
                max_chain_len = self.max_chain_len
                DOP = estimate_max_DOP(monomer_smarts, max_chain_len)
            
            if max_chain_len > self.chain_len_limit:
                raise ExcessiveChainLengthError(f'Cannot create reduction with over {self.chain_len_limit} atoms (requested {max_chain_len})')
            
            chain = build_linear_polymer(monomer_smarts, DOP=DOP, reverse_term_labels=(polymer.mol_name in self.flip_term_labels))
            chain.save(str(self.struct_output/f'{polymer.mol_name}.pdb'), overwrite=True)
            copyfile(polymer.monomer_file_uncharged, self.mono_output/f'{polymer.mol_name}.json')

        return polymer_fn
    
class ChargeAssignment(WorkflowComponent):
    desc = 'Partial charge assignment'
    name = 'charge'

    def __init__(self, chg_params : ChargingParameters, **kwargs):
        '''Load charging parameters from central resource file'''
        self.chg_params = chg_params

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        parser.add_argument('-cp', '--chg_params_name', help=f'Name of the charging parameters preset file to load for charging (available files are {avail_chg_templates})', required=True)
        parser.add_argument('-dir', '--directory', help='Path of the folder in which the chosen charging parameters preset file resides', type=Path, default=impres.files(resources.chg_templates))

    @classmethod
    def process_argparse_args(self, args: Namespace) -> dict[Any, Any]:
        '''Load ChargingParameters from files specified in CLI args'''
        return {
            'chg_params' :  ChargingParameters.from_file( default_suffix(args.directory / args.chg_params_name, suffix='json') )
        }

    # TOSELF : overwrite / charge status force in assert_filter_prefs?

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Ensure a Polymer has all partial charge sets'''
            assert(polymer.has_monomer_info)

            # 1) ENSURING CHARGES AND RELATED FILES FOR ALL BASE CHARGING METHODS EXIST
            for chg_method in self.chg_params.charge_methods:
                chgr = CHARGER_REGISTRY[chg_method]()
                polymer.assert_charges_for(chgr, return_cmol=False)
                poly_logger.info('') # add gaps between charge method for breathing room
                
            # 2) GENERATE RESIDUE-AVERAGED LIBRARY CHARGES FROM THE CHARGE SET OF CHOICE
            if polymer.has_monomer_info_charged: # load precomputed residue charges if existing set is found
                poly_logger.info(f'Found charged JSON, loading averaged charges for {polymer.mol_name} residues from file')
                residue_charges = polymer.monomer_info_charged.charges
            else: # otherwise, recompute them from the charge set of choice
                poly_logger.info(f'Averaging {self.chg_params.averaging_charge_method} charges over {polymer.mol_name} residues')
                residue_charges = get_averaged_residue_charges(
                    cmol=polymer.charged_offmol(self.chg_params.averaging_charge_method), # TODO : add check to ensure this ISN'T and averaging method
                    monomer_info=polymer.monomer_info_uncharged
                )
            
            if (not polymer.has_monomer_info_charged) or self.chg_params.overwrite_chg_mono: # cast charges residues to file if none exist or is explicitly overwriting
                polymer.create_charged_monomer_file(residue_charges) # TOSELF : this is separate from the above clauses as it might be called regardless of existing charges during overwrite

            avg_chgr = AveragingCharger() # generate precomputed charge set for full molecule from residue library charges
            avg_chgr.set_residue_charges(residue_charges)
            polymer.assert_charges_for(avg_chgr, return_cmol=False)
        
        return polymer_fn

class RunSimulations(WorkflowComponent):
    desc = 'Prepares and integrates MD simulation for chosen molecules in OpenMM'
    name = 'simulate'

    def __init__(self, sim_params : Union[SimulationParameters, Iterable[SimulationParameters]], sequential : bool=False, **kwargs):
        '''Initialize 1 or more sets of simulation parameters'''
        if not sim_params:
            raise ValueError('Must specify at least 1 simulation parameter preset')
        
        self.sim_params = asiterable(sim_params) # handle singleton case
        self.sequential = sequential

        if self.sequential: # perform precheck to ensure sequential simulation load is possible prior to execution
            for sim_params_indiv in self.sim_params: 
                if not sim_params_indiv.save_state:
                    raise AttributeError(f'Cannot include {sim_params_indiv.affix} in chained sequence, as it reports Checkpoint (not State)')

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-sp', '--sim_param_names', help=f'Name of the simulation parameters preset file(s) to load for simulation (available files are {avail_sim_templates})', action='store', nargs='+', required=True)
        parser.add_argument('-dir', '--directory'     , help='Path of the folder in which the chosen simulation parameters preset file(s) reside', type=Path, default=impres.files(resources.sim_templates))
        parser.add_argument('-seq', '--sequential'    , help='If set and multiple parameter sets are passed, each simulation will be initialized with the final snapshot of the previous simulation', action='store_true')

    @classmethod
    def process_argparse_args(self, args: Namespace) -> dict[Any, Any]:
        '''Load SimulationParameter sets from the list of names and directory specified'''
        return {
            'sim_params' : [
                SimulationParameters.from_file(default_suffix(args.directory / sim_param_name, suffix='json')) 
                    for sim_param_name in args.sim_param_names
            ],
            'sequential' : args.sequential
        }

    def assert_filter_prefs(self, molbuf : MolFilterBuffer) -> list[MolFilter]:
        '''Assert any additional preferences for filters beyond the default molecule filters'''
        molbuf.charges = True # force preference for charged molecules (can't run simulations otherwise)
        return molbuf.filters

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Run OpenMM simulation(s) according to sets of predefined simulation parameters'''
            N = len(self.sim_params)
            for i, sim_params_indiv in enumerate(self.sim_params):
                poly_logger.info(f'Running simulation {i + 1} / {N} ("{sim_params_indiv.affix}")')

                # initialize OpenFF Interchange from Molecule
                interchange = polymer.interchange(
                    forcefield_path=sim_params_indiv.forcefield_path,
                    charge_method=sim_params_indiv.charge_method,
                    periodic=sim_params_indiv.periodic
                )

                # Create ensemble-specific Simulation from Interchange 
                sim_factory = EnsembleSimulationFactory.registry[sim_params_indiv.ensemble.upper()]() # case-insensitive check for simulation creators for the desired ensemble
                simulation = sim_factory.create_simulation(interchange, sim_params=sim_params_indiv)
                if (self.sequential) and (i != 0): # if sequential mode is enabled and the current simulation is not the first...
                    poly_logger.info(f'Loading initial configuration from previous simulation "{polymer.newest_sim_dir.name}"')
                    sim_paths, prev_sim_params = polymer.load_sim_paths_and_params(sim_dir=polymer.newest_sim_dir) # ...load the paths from the most recent simulation (i.e. the previous one in the batch) !NOTE! - CRITICAL that sim params are discarded here
                    
                    if prev_sim_params.save_state: # ... and instantiate the current simulation from its checkpoint file (must be converted to str, since OpenMM can't handle Path objects)
                        simulation.loadState(str(sim_paths.checkpoint)) 
                    else:
                        simulation.loadCheckpoint(str(sim_paths.checkpoint)) 

                # Create output folder, populate with simulation files, and integrate
                sim_folder = polymer.make_sim_dir(affix=sim_params_indiv.affix)
                run_simulation(simulation, sim_params=sim_params_indiv, output_folder=sim_folder, output_name=polymer.mol_name)
                poly_logger.info('') # whitespace to provide breathing room

        return polymer_fn

class TransferMonomers(WorkflowComponent):
    desc = 'Transfer residue-averaged charged monomer files between Polymers in two collections'
    name = 'transfer_mono'

    def __init__(self, target_mgr : PolymerManager, **kwargs):
        self.target_mgr = target_mgr

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-targ', '--target_path', help='The path to the target output collection of Polymers to move charged monomers to', type=Path, required=True)

    @classmethod
    def process_argparse_args(self, args: Namespace) -> dict[Any, Any]:
        return {'target_mgr' : PolymerManager(args.target_path)}

    def assert_filter_prefs(self, molbuf : MolFilterBuffer) -> list[MolFilter]:
        '''Assert any additional preferences for filters beyond the default molecule filters'''
        molbuf.charges = True # force preference for charges 
        mol_filters = molbuf.filters
        mol_filters.append(has_monomers_chgd) # also assert that, not only do charges exist, but that they've been monomer-averaged

        return mol_filters

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Copies charged monomer files FROM the current Polymer TO to a corresponding Polymer in the target collection'''
            counterpart = self.target_mgr.polymers[polymer.mol_name] 
            assert(polymer.solvent == counterpart.solvent)

            if not counterpart.has_monomer_info_uncharged:
                polymer.transfer_file_attr('monomer_file_uncharged', counterpart)
            
            if not counterpart.has_monomer_info_charged:
                polymer.transfer_file_attr('monomer_file_charged', counterpart)
        
        return polymer_fn
    
class TrajectoryAnalysis(WorkflowComponent):
    desc = 'Computes RDF and property time series data and saving to csvs for plotting and analysis'
    name = 'analyze'

    def __init__(self, sim_time : Optional[int], traj_sample_interval : int=1, **kwargs):
        '''Defining simulation-based filters'''
        self.sim_dir_filters = [has_binary_traj]
        if sim_time is not None:
            is_long_sim = lambda sim_paths, sim_params : (sim_params.total_time == sim_time*nanosecond)
            self.sim_dir_filters.append(is_long_sim)
        
        self.traj_sample_interval = traj_sample_interval

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-t', '--sim_time'              , help='If set, will only analyze trajectories run for this number of nanoseconds', action='store', type=float)
        parser.add_argument('-tsi', '--traj_sample_interval', help='How often to sample trajectory frames when loading (equilvalent to "stride" in mdtraj); useful for huge trajectories', action='store', type=int, default=1)

    def assert_filter_prefs(self, molbuf : MolFilterBuffer) -> list[MolFilter]:
        '''Assert any additional preferences for filters beyond the default molecule filters'''
        mol_filters = molbuf.filters
        mol_filters.append(has_sims)

        return mol_filters

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Analyze trajectories to obtain polymer property data in reusable CSV form'''
            sim_dirs_to_analyze = polymer.filter_sim_dirs(conditions=self.sim_dir_filters)
            N = len(sim_dirs_to_analyze)

            for i, (sim_dir, (sim_paths, sim_params)) in enumerate(sim_dirs_to_analyze.items()):
                poly_logger.info(f'Found trajectory {sim_paths.trajectory} ({i + 1}/{N})')
                traj = polymer.load_traj(sim_dir)

                # save and plot RDF data
                poly_logger.info('Calculating pairwise radial distribution functions')
                rdf_dataframe = trajectory.acquire_rdfs(traj, max_rad=1.0*nanometer)
                rdf_save_path = sim_dir/'rdfs.csv'
                sim_paths.spatial_data = rdf_save_path
                rdf_dataframe.to_csv(rdf_save_path, index=False)

                # save and plot property data
                poly_logger.info('Calculating polymer shape properties')
                prop_dataframe = trajectory.acquire_time_props(traj, time_points=sim_params.time_points[::self.traj_sample_interval]) 
                prop_save_path = sim_dir/'time_series.csv'
                sim_paths.time_data = prop_save_path
                prop_dataframe.to_csv(prop_save_path, index=False)

                sim_paths.to_file(polymer.simulation_paths[sim_dir]) # update references to analyzed data files in path file
                poly_logger.info(f'Successfully exported trajectory analysis data')
                poly_logger.info('') # whitespace to provide breathing room
            
        return polymer_fn
    
class TransferSimSnapshot(WorkflowComponent): # TODO : decompose this into cloning, sim (already implemented), and structure transfer components
    desc = 'Take the final frame of a simulation and generate a new Polymer with that frame as its PDB structure'
    name = 'transfer_struct'

    def __init__(self, dest_dir : Optional[Path]=None, snapshot_idx : int=-1, clone_affix : str='clone', sim_dir_filters : Optional[Iterable[SimDirFilter]]=None, **kwargs):
        '''Define parameters for vacuum anneal, along with number of new conformers'''
        self.dest_dir = dest_dir
        self.snapshot_idx = snapshot_idx
        self.clone_affix = clone_affix

        if sim_dir_filters is None:
            sim_dir_filters = []
        self.sim_dir_filters = sim_dir_filters

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-dest', '--dest_dir'   , help='Destination directory into which the structure-altered clone should be sent', type=Path)
        parser.add_argument('-sid', '--snapshot_idx', help='Index of the frame of the vacuum anneal simulation to take as the new conformation (by default -1, i.e. the final frame)', type=int, default=-1)
        parser.add_argument('-caf', '--clone_affix' , help='An additional descriptive string to add to the name of the resulting clone', default='clone')

    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Run quick vacuum NVT sim at high T for specified number of runs to generate perturbed starting structures'''
            # generate clone to anneal
            parent_dir = self.dest_dir / polymer.base_mol_name
            parent_dir.mkdir(exist_ok=True)

            conf_clone = polymer.clone(
                dest_dir=parent_dir,
                clone_affix=self.clone_affix,
                clone_solvent=True,
                clone_structures=True,
                clone_monomers=True,
                clone_charges=True,
                clone_sims=False
            )
            
            # replace clone's starting structure with new anneal structure
            poly_logger.info('Extracting final conformation from simulation')
            traj = polymer.load_traj(polymer.newest_sim_dir)
            new_conf = traj[self.snapshot_idx]
            poly_logger.info('Applying new conformation to clone')
            new_conf.save(conf_clone.structure_file) # overwrite the clone's structure with the new conformer
            
        return polymer_fn

class _SlurmSbatch(WorkflowComponent):
    desc = 'Private component for handling parallel dispatch of single-polymer job submissions to slurm'
    name = '_sbatch'

    def __init__(self, component : WorkflowComponent, sbatch_script : Path, python_script_name : str, source_path : Path, jobtime : datetime.time, collect_job_ids : bool=False, **kwargs):
        self.component = component
        self.sbatch_script = sbatch_script
        self.python_script_name = python_script_name
        self.source_path = source_path
        
        self.jobtime = jobtime
        self.job_ids = []
        self.collect_job_ids = collect_job_ids

    @staticmethod
    def argparse_inject(parser : ArgumentParser) -> None:
        '''Flexible support for instantiating addition to argparse in an existing script'''
        parser.add_argument('-comp', '--component_name'  , help='Name of the target WorkflowComponent type to parallelize', required=True)
        parser.add_argument('-src', '--source_path'      , help='The Path to the target collection of Polymers', type=Path, required=True)
        parser.add_argument('-sb', '--sbatch_script'     , help='Name of the target slurm job script to use for submission', default='slurm_dispatch.job', type=Path)
        parser.add_argument('-py', '--python_script_name', help='Name of the Python script which should be called on')
        parser.add_argument('-jid', '--collect_job_ids'  , help='Whether or not to gather job IDs when submitting (useful for creating dependencies in serial workflows)', action='store_true')

        parser.add_argument('-hr', '--hours'   , help='Number of hours to allocate in job walltime'  , type=int, default=0)
        parser.add_argument('-min', '--minutes', help='Number of minutes to allocate in job walltime', type=int, default=30)
        parser.add_argument('-sec', '--seconds', help='Number of seconds to allocate in job walltime', type=int, default=0)

    @classmethod
    def process_argparse_args(self, args: Namespace) -> dict[Any, Any]:
        return {
            'component' : WorkflowComponent.registry[args.component_name],
            'sbatch_script' : default_suffix(args.sbatch_script, suffix='job'),
            'python_script_name' : args.python_script_name,
            'source_path' : args.source_path,
            'collect_job_ids' : args.collect_job_ids,
            'jobtime' : datetime.time(hour=args.hours, minute=args.minutes, second=args.seconds)
        }
    
    # methods UNIQUE to this component
    @property
    def jobtime_str(self) -> str:
        '''Format the specified job time into a string passable to sbatch'''
        return self.jobtime.strftime('%H:%M:%S')
    
    @property
    def dependency_str(self) -> str:
        '''String of job ids which can be passed to subsequent sbatch job dependency arg (as after<arg>:self.dependency_str)'''
        return ':'.join(self.job_ids)

    @staticmethod
    def extract_job_id(shell_str : Union[str, bytes]) -> str:
        '''Extracts job id from shell-echoed string after slurm job submission'''
        # JOB_ID_RE = re.compile(r'Submitted batch job (\d+)')
        JOB_ID_RE = re.compile(r'(\d+)')
        if isinstance(shell_str, bytes):
            shell_str = shell_str.decode(encoding='utf-8')

        return re.search(JOB_ID_RE, str(shell_str)).groups()[0]
    
    def fetch_variable_args(self) -> str:
        '''Collect the arguments passed to the Component for propogation to individual job calls'''
        arg_start_idx = sys.argv.index(self.component.name) + 1 # find where Component-specific arguments begin (immediately after job type spec)
        arg_str = ' '.join(sys.argv[arg_start_idx:])       # collate into space-delimited string

        return arg_str

    def generate_sbatch_cmd(self, mol_name : str) -> str: 
        '''Generate an sbatch command call string to be executed as a subprocess'''
        jobname = f'{mol_name}_{self.component.name}_dispatch'
        
        return' '.join([
            'sbatch',
            '--parsable', # job submission only returns job ID (easier to parse)
            f'--time {self.jobtime_str}',
            f'--job-name {jobname}',
            f'--output slurm_logs/{jobname}.log',
            str(self.sbatch_script), # define target .job script
            self.python_script_name, # script arguments begin here
            str(self.source_path), 
            self.component.name,
            mol_name,
            self.fetch_variable_args()
        ])

    # polymer dispatch generation (NOT unique here)
    def make_polymer_fn(self) -> PolymerFunction:
        '''Create wrapper for handling in logger'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Echo requisite information for a single molecule'''
            slurm_out = subprocess.check_output([self.generate_sbatch_cmd(polymer.mol_name)], shell=True) # submit job via subprocess shell call
            if self.collect_job_ids:
                self.job_ids.append(self.extract_job_id(slurm_out))
                # self.job_ids.append( slurm_out.decode('utf-8') ) # subprocess return bytes instead of str for some reason
        
        return polymer_fn