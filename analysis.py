# Custom imports
from polymer_utils.graphics.plotutils import presize_subplots
from polymer_utils.general import hasunits

# General imports
from pathlib import Path
from itertools import combinations

# Numeric processing and plotting
import mdtraj as mdt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Typing and Subclassing
from typing import Any, Callable
from dataclasses import dataclass, field

# Units
from openmm.unit import Unit, Quantity  # purely for typehints
from openmm.unit import nanometer, dimensionless
class MissingUnitsError(Exception):
    pass

# Plotting properties
@dataclass
class PolyProp:
    '''For encapsulating polymer property data and labels when plotting in series'''
    calc   : Callable
    name   : str
    abbr   : str
    unit   : Unit
    kwargs : dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f'{self.name} ({self.abbr}, {self.unit.get_symbol()})'

    def compute(self, traj : mdt.Trajectory) -> Any:
        '''Apply method to a trajectory'''
        return self.calc(traj, **self.kwargs)
    
DEFAULT_PROPS = [ # the base properties of interest for 
    PolyProp(calc=mdt.compute_rg                , name='Radius of Gyration'             , abbr='Rg'  , unit=nanometer),
    PolyProp(calc=mdt.shrake_rupley             , name='Solvent Accessible Surface Area', abbr='SASA', unit=nanometer**2, kwargs={'mode' : 'residue'}),
    PolyProp(calc=mdt.relative_shape_antisotropy, name='Relative Shape Anisotropy'      , abbr='K2'  , unit=dimensionless)    
]


# Data output functions (mediated via DataFrames)
def acquire_rdfs(traj : mdt.Trajectory, max_rad : Quantity=1*nanometer) -> pd.DataFrame:
    '''
    Takes a Trajectory and a Dataframe and writes columns to the DataFrame for all possible pairwise Radial Distribution Functions,
    along with the radii sampled up to the specified maximum radius (must have units!)
    
    Optionally, can also return Figure and Axes with the RDFs plotted
    '''
    if not hasunits(max_rad): 
        raise MissingUnitsError
    max_rad = max_rad.in_units_of(nanometer) # assert that distances will be measured in nm
    rad_unit = max_rad.unit.get_symbol()

    elem_types = set(atom.element.symbol for atom in traj.topology.atoms)
    elem_pairs = list(combinations(elem_types, 2)) # every possible choice of 2 distinct atom types

    out_dframe = pd.DataFrame()
    for (elem1, elem2) in elem_pairs:
        atom_id_pairs = traj.topology.select_pairs(f'element {elem1}', f'element {elem2}')
        rad_points, rdf = mdt.compute_rdf(traj, pairs=atom_id_pairs, r_range=(0.0, max_rad._value))
        out_dframe[f'Radius ({rad_unit})'] = rad_points # NOTE : radii range will be same for all pairs, so overwrite beyond first RDF is OK
        out_dframe[f'g(r) ({elem1}-{elem2})'] = rdf 

    return out_dframe

def acquire_time_props(traj : mdt.Trajectory, properties : list[PolyProp], time_points : np.ndarray=None) -> pd.DataFrame:
    '''Compute and plot a battery of labelled and unit-ed properties over a given trajectory'''
    out_dframe = pd.DataFrame()
    if time_points is not None:
        unit_str = f' ({time_points.unit.get_symbol()})' if hasunits(time_points) else ''
        out_dframe[f'Sample Time{unit_str}'] = time_points

    for prop in properties:
        time_series = prop.compute(traj) 
        out_dframe[prop.label] = time_series * prop.unit

    return out_dframe


# Property plotting functions
def plot_rdfs(dframe : pd.DataFrame, **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    '''Takes a DataFrame populated with RDF data and generates sequential plots'''
    radii = dframe.filter(regex='Radius')
    radii_label = radii.columns[0]
    rdfs  = dframe.filter(regex='g\(r\)') # need escapes to treat parens as literals

    fig, ax = presize_subplots(nrows=1, ncols=len(rdfs.columns), **plot_kwargs)
    for curr_ax, (name, rdf) in zip(ax.flatten(), rdfs.items()):
        curr_ax.plot(radii, rdf)
        curr_ax.set_xlabel(radii_label)
        curr_ax.set_ylabel(name)

    return fig, ax

def plot_time_props(dframe : pd.DataFrame, **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    '''Takes a DataFrame populated with polymer time series property data and generates sequential plots'''
    times     = dframe.filter(regex='Time')
    times_label = times.columns[0]
    prop_data = dframe[[label for label in dframe.columns if label != times_label]] # index props with all non-time point columns

    fig, ax = presize_subplots(nrows=1, ncols=len(prop_data.columns), **plot_kwargs)
    for curr_ax, (name, rdf) in zip(ax.flatten(), prop_data.items()):
        curr_ax.plot(times, rdf)
        curr_ax.set_xlabel(times_label)
        curr_ax.set_ylabel(name)

    return fig, ax


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