# Cusotm imports
from polymer_utils.graphics.plotutils import presize_subplots

# General imports
from pathlib import Path
from itertools import combinations

# Numeric processing and plotting
import mdtraj as mdt
import numpy as np
import matplotlib.pyplot as plt

# Typing and Subclassing
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

# Units
from openmm.unit import Unit, nanometer # purely for typehints


# Plotting properties
@dataclass
class PolyProp:
    '''For encapsulating polymer property data and labels when plotting in series'''
    calc   : Callable
    name   : str
    abbr   : str
    unit   : Unit
    kwargs : dict = field(default_factory=dict)

    def compute(self, traj : mdt.Trajectory):
        '''Apply method to a trajectory'''
        return self.calc(traj, **self.kwargs)

def plot_rdfs(traj : mdt.Trajectory, rad_unit : Unit=nanometer, header : str='', save_path : Path=None) -> dict[str, np.array]:
    '''Plot pairwise radial distributions functions for all possible pairs of elements in a simulation trajectory'''
    elem_types = set(atom.element.symbol for atom in traj.topology.atoms)
    fig, ax = presize_subplots(nrows=1, ncols=len(elem_types), scale=15.0)
    fig.suptitle(header, fontsize=20)

    rdfs = {}
    for curr_ax, (elem1, elem2) in zip(ax.flatten(), combinations(elem_types, 2)):
        atom_id_pairs = traj.topology.select_pairs(f'element {elem1}', f'element {elem2}')
        rdf = mdt.compute_rdf(traj, pairs=atom_id_pairs, r_range=(0.0, 1.0))
        rdfs[f'{elem1}{elem2}'] = rdf # save result for output

        curr_ax.plot(*rdf)
        curr_ax.set_xlabel(f'Radial distance ({rad_unit.get_symbol()})')
        curr_ax.set_ylabel(f'g(r) - {elem1} vs {elem2}')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return rdfs

def plot_poly_props(traj : mdt.Trajectory, properties : list[PolyProp], sample_interval : int, header : str='', save_path : Path=None) -> dict[str, np.array]:
    '''Compute and plot a battery of labelled and unit-ed properties over a given trajectory'''
    frame_nums = np.arange(0, traj.n_frames * sample_interval, step=sample_interval)
    fig, ax = presize_subplots(nrows=1, ncols=len(properties))
    fig.suptitle(header, fontsize=20)

    prop_data = {}
    for curr_ax, prop in zip(ax.flatten(), properties):
        time_series = prop.compute(traj) 
        prop_data[prop.abbr] = time_series * prop.unit

        curr_ax.plot(frame_nums, time_series)
        curr_ax.set_xlabel('Trajectory frame')
        curr_ax.set_ylabel(f'{prop.name} ({prop.unit.get_symbol()})')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return prop_data

# Custom implementations of polymer property calculations - recommend using MDTraj or similar implementations first
def compute_gyration_tensor(coords : np.ndarray, use_eins=True) -> np.ndarray:
    '''Determines the gyration tensor of a set of atomic coordinates (and its diagonalization)
    Expects coordinates in an Nx3 array, with each column representing the x, y, and z coordinates, respectively'''
    N, ncols = arr_shape = coords.shape
    if ncols != 3:
        raise ValueError(f'Coordinate array must be of shape Nx3, not {arr_shape}')

    COM = coords.mean(axis=0) # centre-of-mass coord
    rs = coords - COM # place origin at center of mass

    if use_eins: # use Einstein summation for notational and memory compactness
        gyr_tens = np.einsum('ij, ik->jk', rs, rs) / N 
    else: # use direct displacement summation, less efficient but easier to understand
        gyr_tens = np.zeros((3, 3))
        for point in rs:
            S_i = np.outer(point, point)
            gyr_tens += S_i
        gyr_tens /= N

    return gyr_tens

def diagonalize(matrix : np.ndarray) -> list[np.ndarray]:
    '''Performs eigendecomposition of a matrix
    Returns the basis and diagonal matrices as a list [P, D, P^-1]''' 
    eivals, eivecs = np.linalg.eig(matrix) # perform eigendecomposition to obtain principle components
    return [eivecs, np.diag(eivals), np.linalg.inv(eivecs)] # return vector containing P, D, and P^-1

def compute_Rg_and_K2(gyr_tens : np.ndarray, use_diag : bool=False) -> tuple[float, float]:
    '''Determine the radius of gyration (Rg) and radial shape anisotropy (K2) from a molecules gyration tensor'''
    if use_diag:
        P, D, P_inv = diagonalize(gyr_tens)
        I1 = np.trace(D)
        I2 = (I1**2 - np.trace(D**2)) / 2
    else:
        S, _eivecs = np.linalg.eig(gyr_tens)
        I1 = S.sum()
        I2 = (I1**2 - np.sum(S**2)) / 2

    Rg = np.sqrt(I1)
    K2 = 1 - 3*(I2 / I1**2)
    
    return Rg, K2