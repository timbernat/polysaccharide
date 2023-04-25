from typing import Any, Callable
from dataclasses import dataclass, field

import numpy as np
from mdtraj import Trajectory

from openmm.unit import Unit
from openmm.unit import nanometer, dimensionless


@dataclass
class PolyProp:
    '''For encapsulating polymer property calculation methods, labels, and plotting info'''
    calc   : Callable
    name   : str
    abbr   : str
    unit   : Unit
    kwargs : dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f'{self.name} ({self.abbr}, {self.unit.get_symbol()})'

    def compute(self, traj : Trajectory) -> Any:
        '''Apply method to a trajectory'''
        return self.calc(traj, **self.kwargs)
    
import mdtraj as mdt

DEFAULT_PROPS = [ # the base properties of interest for the 2023 monomer spec study
    PolyProp(calc=mdt.compute_rg                , name='Radius of Gyration'             , abbr='Rg'  , unit=nanometer),
    PolyProp(calc=mdt.shrake_rupley             , name='Solvent Accessible Surface Area', abbr='SASA', unit=nanometer**2, kwargs={'mode' : 'residue'}),
    PolyProp(calc=mdt.relative_shape_antisotropy, name='Relative Shape Anisotropy'      , abbr='K2'  , unit=dimensionless)    
]


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