'''For executing prepared MD OpenMM simulations'''

# General
from pathlib import Path

# Typing and subclassing
# from typing import Union

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# OpenForceField
from openff.interchange import Interchange

# Custom imports
from . import preparation
from .records import SimulationParameters
from .ensemble import ENSEMBLE_REGISTRY


# Functions for actually running simulations
def run_simulation(interchange : Interchange, sim_params : SimulationParameters, output_folder : Path, output_name : str, ensemble : str='NPT') -> None:
    '''
    Initializes an OpenMM simulation from a SMIRNOFF Interchange in the desired ensemble
    Creates relevant simulation files, generates Reporters for state, checkpoint, and trajectory data,
     performs energy minimization, then integrates the trajectory for the desired number of steps
    '''
    sim_factory = ENSEMBLE_REGISTRY[ensemble.upper()]() # case-insensitive check for simulation creators for the desired ensemble
    simulation = sim_factory.create_simulation(interchange, sim_params=sim_params)

    sim_paths = preparation.prepare_simulation_paths(output_folder, output_name, report_to_pdb=sim_params.report_to_pdb)
    reporters = preparation.prepare_simulation_reporters(sim_paths, sim_params)
    sim_params.to_file(sim_paths.sim_params) # TOSELF : this is not a parameters checkpoint file UPDATE, but rather the initial CREATION of the checkpoint file
    preparation.config_simulation(simulation, reporters, checkpoint_path=sim_paths.checkpoint)

    LOGGER.info('Performing energy minimization')
    simulation.minimizeEnergy()

    LOGGER.info(f'Integrating {sim_params.total_time} OpenMM sim at {sim_params.temperature} and {sim_params.pressure} for {sim_params.num_steps} steps')
    simulation.step(sim_params.num_steps)