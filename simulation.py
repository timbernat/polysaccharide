# General
from pathlib import Path
import re
import numpy as np

# Typing and subclassing
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from .filetree import JSONSerializable, JSONifiable

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# OpenForceField
from openff.units import unit as offunit # need OpenFF version of unit for Interchange positions for some reason
from openff.interchange import Interchange

from openmm import Integrator
from openmm.openmm import Force
from openmm.app import Simulation
from openmm.app import PDBReporter, StateDataReporter, CheckpointReporter
Reporter = Union[PDBReporter, StateDataReporter, CheckpointReporter] # for clearer typehinting

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

def run_simulation(simulation : Simulation, output_folder : Path, output_name : str, sim_params : SimulationParameters) -> Path:
    '''Takes a Simulation object, performs energy minimization, and runs simulation for specified number of time steps
    Recording PBD frames and the specified property data to CSV at the specified frequency'''
    # creating paths to requisite output files
    prefix = f'{output_name}{"_" if output_name else ""}'
    sim_paths_out = output_folder / f'{prefix}sim_paths.json'
    sim_paths = SimulationPaths(
        sim_params=output_folder / f'{prefix}sim_parameters.json',
        trajectory=output_folder / f'{prefix}traj.pdb',
        state_data=output_folder / f'{prefix}state_data.csv',
        checkpoint=output_folder / f'{prefix}checkpoint.chk',
    )
    sim_paths.to_file(sim_paths_out)
    sim_params.to_file(sim_paths.sim_params) 

    # for saving pdb frames and reporting state/energy data - NOTE : all file paths must be stringified for OpenMM
    pdb_rep = PDBReporter(str(sim_paths.trajectory), sim_params.record_freq)  # save frames at the specified interval
    state_rep = StateDataReporter(str(sim_paths.state_data), sim_params.record_freq, **sim_params.reported_state_data)
    for rep in (pdb_rep, state_rep):
        simulation.reporters.append(rep) # add any desired reporters to simulation for tracking

    # minimize and run simulation
    simulation.minimizeEnergy()
    simulation.saveCheckpoint(str(sim_paths.checkpoint)) # save initial minimal state to simplify reloading process
    LOGGER.info(f'Running {sim_params.total_time} OpenMM sim at {sim_params.temperature} and {sim_params.pressure} for {sim_params.num_steps} steps')
    simulation.step(sim_params.num_steps)

    return sim_paths_out
    