# General
from pathlib import Path

# OpenForceField
from openff.units import unit
from openff.interchange import Interchange

from openmm import Integrator
from openmm.app import Simulation, PDBReporter, StateDataReporter
from openmm.unit import nanometer


def create_simulation(interchange : Interchange, integrator : Integrator) -> Simulation:
    '''Specifies configuration for an OpenMM Simulation - Interchange load alows many routes for creation'''
    openmm_sys = interchange.to_openmm(combine_nonbonded_forces=True) 
    openmm_top = interchange.topology.to_openmm()
    openmm_pos = interchange.positions.m_as(unit.nanometer) * nanometer

    simulation = Simulation(openmm_top, openmm_sys, integrator)
    simulation.context.setPositions(openmm_pos)

    return simulation

def run_simulation(simulation : Simulation, output_folder : Path, output_name : str, num_steps : int, record_freq : int) -> None:
    '''Takes a Simulation object, performs energy minimization, and runs simulation for specified number of time steps
    Recording PBD frames and numerical data to file at the specified frequency'''
    folder_name = str(output_folder) # for some reason OpenMM simulations don't like Path objects (only take strings)

    # for saving pdb frames and reporting state/energy data
    pdb_rep = PDBReporter(f'{folder_name}/{output_name}_frames.pdb', record_freq)  # save frames at the specified interval
    state_rep = StateDataReporter(f'{folder_name}/{output_name}_data.csv', record_freq, step=True, potentialEnergy=True, temperature=True)
    reporters = (pdb_rep, state_rep)
    for rep in reporters:
        simulation.reporters.append(rep) # add any desired reporters to simulation for tracking

    # minimize and run simulation
    simulation.minimizeEnergy()
    simulation.saveCheckpoint(f'{folder_name}/{output_name}_checkpoint.chk') # save initial minimal state to simplify reloading process
    simulation.step(num_steps)