'''Biolerplate for setting up OpenMM Simulations and related files'''

# General
from pathlib import Path

# Typing and subclassing
from typing import Iterable, Optional, Union

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# OpenForceField
from openff.units import unit as offunit # need OpenFF version of unit for Interchange positions for some reason
from openff.interchange import Interchange

from openmm import Integrator
from openmm.openmm import Force
from openmm.app import Simulation

from openmm.app import PDBReporter, DCDReporter, StateDataReporter, CheckpointReporter
Reporter = Union[PDBReporter, DCDReporter, StateDataReporter, CheckpointReporter] # for clearer typehinting

# Custom Imports
from .records import SimulationPaths, SimulationParameters

# Units, quantities, and dimensions
from openmm.unit import nanometer


# Functions for creating Simulation (and related) objects
def create_simulation(interchange : Interchange, integrator : Integrator, forces : Optional[Iterable[Force]]=None) -> Simulation:
    '''Specifies configuration for an OpenMM Simulation - Interchange load alows many routes for creation'''
    openmm_sys = interchange.to_openmm(combine_nonbonded_forces=True) 
    openmm_top = interchange.topology.to_openmm()
    openmm_pos = interchange.positions.m_as(offunit.nanometer) * nanometer

    if forces: # deliberately sparse to handle both Nonetype and empty list
        for force in forces: 
            openmm_sys.addForce(force)

    LOGGER.info('Creating OpenMM Simulation from Interchange')
    simulation = Simulation(openmm_top, openmm_sys, integrator)
    simulation.context.setPositions(openmm_pos)

    return simulation

def prepare_simulation_paths(output_folder : Path, output_name : str, sim_params : SimulationParameters) -> SimulationPaths:
    '''Takes a Simulation object, performs energy minimization, and runs simulation for specified number of time steps
    Recording PDB frames and the specified property data to CSV at the specified frequency'''
    # creating paths to requisite output files
    prefix = f'{output_name}{"_" if output_name else ""}'
    sim_paths_out = output_folder / f'{prefix}sim_paths.json'
    sim_paths = SimulationPaths(
        sim_params=output_folder / f'{prefix}sim_parameters.json',
        trajectory=output_folder / f'{prefix}traj.{"dcd" if sim_params.binary_traj else "pdb"}',
        state_data=output_folder / f'{prefix}state_data.csv',
        checkpoint=output_folder / f'{prefix}checkpoint.{"xml" if sim_params.save_state else "chk"}',
    )
    sim_paths.to_file(sim_paths_out)
    LOGGER.info(f'Generated simulation record files at {sim_paths_out}')

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
    check_rep = CheckpointReporter(str(sim_paths.checkpoint), reportInterval=sim_params.record_freq, writeState=sim_params.save_state)
    state_rep = StateDataReporter(str(sim_paths.state_data), reportInterval=sim_params.record_freq, **sim_params.reported_state_data)

    return (traj_rep, check_rep, state_rep)

def config_simulation(simulation : Simulation, reporters : Iterable[Reporter], checkpoint_path : Optional[Path]=None) -> None:
    '''Takes a Simulation object, adds data Reporters, saves an initial checkpoint, and performs energy minimization'''
    for rep in reporters:
        simulation.reporters.append(rep) # add any desired reporters to simulation for tracking

    if checkpoint_path is not None:
        simulation.saveCheckpoint(str(checkpoint_path)) # save initial minimal state to simplify reloading process
        LOGGER.info(f'Saved simulation checkpoint at {checkpoint_path}')
