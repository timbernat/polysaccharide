'''For preparing, executing, documenting, and reproducing MD simulations using OpenMM'''

# General
from pathlib import Path
import re
import numpy as np

# Typing and subclassing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union
from .filetree import JSONSerializable, JSONifiable

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# OpenForceField
from openff.units import unit as offunit # need OpenFF version of unit for Interchange positions for some reason
from openff.interchange import Interchange

from openmm import Integrator, VerletIntegrator, LangevinMiddleIntegrator
from openmm.openmm import Force, MonteCarloBarostat
from openmm.app import Simulation

from openmm.app import PDBReporter, DCDReporter, StateDataReporter, CheckpointReporter
Reporter = Union[PDBReporter, DCDReporter, StateDataReporter, CheckpointReporter] # for clearer typehinting

# Units, quantities, and dimensions
import openmm.unit
from openmm.unit import Quantity
from openmm.unit import nanosecond, picosecond, femtosecond
from openmm.unit import atmosphere, kelvin, nanometer


@dataclass
class SimulationParameters(JSONifiable):
    '''For recording the parameters used to run an OpenMM Simulation'''
    total_time  : Quantity
    num_samples : int
    charge_method : str

    report_to_pdb : bool = False
    reported_state_data : dict[str, bool] = field(default_factory=dict)

    timestep    : Quantity = (2 * femtosecond)
    temperature : Quantity = (300 * kelvin)
    pressure    : Quantity = (1 * atmosphere)

    friction_coeff : Quantity = (1 / picosecond)
    barostat_freq : int = 25

    @property
    def num_steps(self) -> int:
        '''Total number of steps in the simulation'''
        return round(self.total_time / self.timestep)
    
    @property
    def record_freq(self) -> int:
        '''Number of steps between each taken sample'''
        return round(self.num_steps / self.num_samples)
    
    @property
    def record_interval(self) -> float:
        '''Length of time between successive samples'''
        return self.record_freq * self.timestep
    
    @property
    def time_points(self) -> np.ndarray[int]:
        '''An array of the time data points represented by the given sampling rate and sim time'''
        return (np.arange(0, self.num_steps, step=self.record_freq) + self.record_freq)* self.timestep # extra offset by recording frequency need to align indices (not 0-indexed)

    # JSON serialization
    @staticmethod
    def serialize_json_dict(unser_jdict : dict[Any, Any]) -> dict[str, JSONSerializable]:
        '''Serialize unit-ful Quantity attrs in a way the JSON can digest'''
        ser_jdict = {}
        for attr_name, attr_val in unser_jdict.items():
            if not isinstance(attr_val, Quantity):
                ser_jdict[attr_name] = attr_val # nominally, copy all non-Quantities directly
            else:
                ser_jdict[f'{attr_name}_value'] = attr_val._value
                ser_jdict[f'{attr_name}_unit' ] = str(attr_val.unit)

        return ser_jdict
    
    @staticmethod
    def unserialize_json_dict(ser_jdict : dict[str, JSONSerializable]) -> dict[Any, Any]:
        '''For unserializing unit-ful Quantities upon load from json file'''
        unser_jdict = {}
        for attr_name, attr_val in ser_jdict.items():
            if attr_name.endswith('_unit'): # skip pure units
                continue 
            elif attr_name.endswith('_value'): # reconstitute Quantities from associated units
                quant_name = re.match('(.*)_value', attr_name).groups()[0] # can't use builtin str.strip, as it removes extra characters
                unit_name = ser_jdict[f'{quant_name}_unit']
                
                if unit_name.startswith('/'): # special case needed to handle inverse units
                    unit = 1 / getattr(openmm.unit, unit_name.strip('/'))
                else:
                    unit = getattr(openmm.unit, unit_name)

                unser_jdict[quant_name] = Quantity(attr_val, unit)
            else: # for non-Quantity entries, load as-is
                unser_jdict[attr_name] = attr_val
        
        return unser_jdict
    
@dataclass
class SimulationPaths(JSONifiable):
    '''Stores paths to various files associated with a completed MD simulation'''
    sim_params : Path
    trajectory : Path
    state_data : Path = None
    checkpoint : Path = None
    
    time_data    : Path = None
    spatial_data : Path = None

    # JSON serialization
    @staticmethod
    def serialize_json_dict(unser_jdict : dict[Any, Any]) -> dict[str, JSONSerializable]:
        '''Convert all Paths to strings'''
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
            if value is not None:
                unser_jdict[key] = Path(value)
            else:
                unser_jdict[key] = value

        return unser_jdict
    

# Functions for creating Simulation (and related) objects
def create_simulation(interchange : Interchange, integrator : Integrator, forces : Optional[list[Force]]=None) -> Simulation:
    '''Specifies configuration for an OpenMM Simulation - Interchange load alows many routes for creation'''
    openmm_sys = interchange.to_openmm(combine_nonbonded_forces=True) 
    openmm_top = interchange.topology.to_openmm()
    openmm_pos = interchange.positions.m_as(offunit.nanometer) * nanometer

    if forces:
        for force in forces:
            openmm_sys.addForce(force)

    LOGGER.info('Creating OpenMM Simulation from Interchange')
    simulation = Simulation(openmm_top, openmm_sys, integrator)
    simulation.context.setPositions(openmm_pos)

    return simulation

def create_simulation_NVE(interchange : Interchange, sim_params : SimulationParameters) -> Simulation:
    '''Initialize and OpenMM Simulation with a Langevin Middle integrator thermostat but NO barostat'''
    integrator  = VerletIntegrator(sim_params.timestep)
    sim = create_simulation(interchange, integrator)
    LOGGER.info('Created NVE Simulation with Verlet Velocity Integrator')
    
    return sim

def create_simulation_NVT(interchange : Interchange, sim_params : SimulationParameters) -> Simulation:
    '''Initialize an OpenMM Simulation with a Langevin Middle integrator thermostat but NO barostat'''
    integrator  = LangevinMiddleIntegrator(sim_params.temperature, sim_params.friction_coeff, sim_params.timestep)
    sim = create_simulation(interchange, integrator)
    LOGGER.info('Created NVT Simulation with Langevin Thermostat')
    
    return sim

def create_simulation_NPT(interchange : Interchange, sim_params : SimulationParameters) -> Simulation:
    '''Initialize and OpenMM Simulation with a Langevin Middle integrator thermostat and Monte Carlo barostat'''
    integrator  = LangevinMiddleIntegrator(sim_params.temperature, sim_params.friction_coeff, sim_params.timestep)
    barostat    = MonteCarloBarostat(sim_params.pressure, sim_params.temperature, sim_params.barostat_freq)
    sim = create_simulation(interchange, integrator, forces=[barostat])
    LOGGER.info('Created NPT Simulation with Langevin Thermostat and Monte Carlo Barostat')
    
    return sim

ENSEMBLE_FACTORIES = {
    'NVE' : create_simulation_NVE,
    'NVT' : create_simulation_NVT,
    'NPT' : create_simulation_NPT
}

# functions for setting up files for accumulating simulation and trajectory data
def prepare_simulation_paths(output_folder : Path, output_name : str, report_to_pdb : bool=False) -> SimulationPaths:
    '''Takes a Simulation object, performs energy minimization, and runs simulation for specified number of time steps
    Recording PBD frames and the specified property data to CSV at the specified frequency'''
    # creating paths to requisite output files
    prefix = f'{output_name}{"_" if output_name else ""}'
    sim_paths_out = output_folder / f'{prefix}sim_paths.json'
    sim_paths = SimulationPaths(
        sim_params=output_folder / f'{prefix}sim_parameters.json',
        trajectory=output_folder / f'{prefix}traj.{"pdb" if report_to_pdb else "dcd"}',
        state_data=output_folder / f'{prefix}state_data.csv',
        checkpoint=output_folder / f'{prefix}checkpoint.chk',
    )
    sim_paths.to_file(sim_paths_out)
    LOGGER.info(f'Created simulation files at {sim_paths_out}')

    return sim_paths

def prepare_simulation_reporters(sim_paths : SimulationPaths, sim_params : SimulationParameters) ->  tuple[Reporter]:
    '''Takes a Simulation object, performs energy minimization, and runs simulation for specified number of time steps
    Recording PBD frames and the specified property data to CSV at the specified frequency'''
    TRAJ_REPORTERS = { # index output formats of reporters by file extension
        '.dcd' : DCDReporter,
        '.pdb' : PDBReporter
    }

    # for saving pdb frames and reporting state/energy data - NOTE : all file paths must be stringified for OpenMM
    TrajReporter = TRAJ_REPORTERS[sim_paths.trajectory.suffix] # look up reporter based on the desired trajectory output file format
    
    traj_rep  = TrajReporter(file=str(sim_paths.trajectory), reportInterval=sim_params.record_freq)  # save frames at the specified interval
    check_rep = CheckpointReporter(str(sim_paths.checkpoint), reportInterval=sim_params.record_freq)
    state_rep = StateDataReporter(str(sim_paths.state_data), reportInterval=sim_params.record_freq, **sim_params.reported_state_data)

    return (traj_rep, check_rep, state_rep)

def config_simulation(simulation : Simulation, reporters : Iterable[Reporter], checkpoint_path : Optional[Path]=None) -> None:
    '''Takes a Simulation object, adds data Reporters, saves an initial checkpoint, and performs energy minimization'''
    for rep in reporters:
        simulation.reporters.append(rep) # add any desired reporters to simulation for tracking

    if checkpoint_path is not None:
        LOGGER.info(f'Saving simulation checkpoint at {checkpoint_path}')
        simulation.saveCheckpoint(str(checkpoint_path)) # save initial minimal state to simplify reloading process

# Functions for actually running simulations
def run_simulation(interchange : Interchange, sim_params : SimulationParameters, output_folder : Path, output_name : str, ensemble : str='NPT') -> None:
    '''
    Initializes an OpenMM simulation from a SMIRNOFF Interchange in the desired ensemble
    Creates relevant simulation files, generates Reporters for state, checkpoint, and trajectory data,
     performs energy minimization, then integrates the trajectory for the desired number of steps
    '''
    sim_factory = ENSEMBLE_FACTORIES[ensemble.upper()] # case-insensitive check for simulation creators for the desired ensemble
    simulation = sim_factory(interchange, sim_params=sim_params)

    sim_paths = prepare_simulation_paths(output_folder, output_name, report_to_pdb=sim_params.report_to_pdb)
    reporters = prepare_simulation_reporters(sim_paths, sim_params)
    sim_params.to_file(sim_paths.sim_params) # TOSELF : this is not a parameters checkpoint file UPDATE, but rather the initial CREATION of the checkpoint file
    config_simulation(simulation, reporters, checkpoint_path=sim_paths.checkpoint)

    LOGGER.info('Performing energy minimization')
    simulation.minimizeEnergy()

    LOGGER.info(f'Integrating {sim_params.total_time} OpenMM sim at {sim_params.temperature} and {sim_params.pressure} for {sim_params.num_steps} steps')
    simulation.step(sim_params.num_steps)

def run_simulation_legacy(simulation : Simulation, output_folder : Path, output_name : str, sim_params : SimulationParameters) -> Path:
    '''Takes a Simulation object, performs energy minimization, and runs simulation for specified number of time steps
    Recording PBD frames and the specified property data to CSV at the specified frequency
    
    Old implementation, Kept in for backwards-compatibility and debug reasons'''
    # creating paths to requisite output files
    prefix = f'{output_name}{"_" if output_name else ""}'
    sim_paths_out = output_folder / f'{prefix}sim_paths.json'
    sim_paths = SimulationPaths(
        sim_params=output_folder / f'{prefix}sim_parameters.json',
        trajectory=output_folder / f'{prefix}traj.{"pdb" if sim_params.report_to_pdb else "dcd"}',
        state_data=output_folder / f'{prefix}state_data.csv',
        checkpoint=output_folder / f'{prefix}checkpoint.chk',
    )
    sim_paths.to_file(sim_paths_out)
    sim_params.to_file(sim_paths.sim_params) # TOSELF : this is not a parameters checkpoint file UPDATE, but rather the initial CREATION of the checkpoint file

    # for saving pdb frames and reporting state/energy data - NOTE : all file paths must be stringified for OpenMM
    check_rep = CheckpointReporter(str(sim_paths.checkpoint), reportInterval=sim_params.record_freq)
    state_rep = StateDataReporter(str(sim_paths.state_data), reportInterval=sim_params.record_freq, **sim_params.reported_state_data)
    if sim_params.report_to_pdb:
        traj_rep = PDBReporter(file=str(sim_paths.trajectory), reportInterval=sim_params.record_freq)  # save frames at the specified interval
    else:
        traj_rep = DCDReporter(file=str(sim_paths.trajectory), reportInterval=sim_params.record_freq)  # save frames at the specified interval

    reporters : tuple[Reporter] =  (check_rep, state_rep, traj_rep)
    for rep in reporters:
        simulation.reporters.append(rep) # add any desired reporters to simulation for tracking

    # minimize and run simulation
    simulation.minimizeEnergy()
    simulation.saveCheckpoint(str(sim_paths.checkpoint)) # save initial minimal state to simplify reloading process
    LOGGER.info(f'Running {sim_params.total_time} OpenMM sim at {sim_params.temperature} and {sim_params.pressure} for {sim_params.num_steps} steps')
    simulation.step(sim_params.num_steps)

    return sim_paths_out
