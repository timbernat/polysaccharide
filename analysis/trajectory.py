# Custom imports
from .polyprops import PolyProp, DEFAULT_PROPS
from ..general import hasunits, MissingUnitsError

# Generic Imports
from itertools import combinations
from pathlib import Path

# Numeric processing and plotting
import mdtraj as mdt
import numpy as np
import pandas as pd

# Units
from openmm.unit import Unit, Quantity  # purely for typehints
from openmm.unit import nanometer, nanosecond


# File I/O
def load_traj(traj_path : Path, topo_path : Path, sample_interval : int=1, remove_solvent : bool=False, inplace : bool=True, **kwargs) -> mdt.Trajectory:
    '''Wrapper to load trajectories from files (avoids mdtraj import in external modules)'''
    traj = mdt.load(traj_path, top=topo_path, stride=sample_interval, **kwargs)
    if remove_solvent:
        traj = traj.remove_solvent(inplace=inplace) # don't generate new copy when de-solvating
    
    return traj

# Data output functions (mediated via DataFrames)
def acquire_rdfs(traj : mdt.Trajectory, max_rad : Quantity=1*nanometer) -> pd.DataFrame:
    '''Takes a Trajectory and produces a DataFrame for all possible pairwise Radial Distribution Functions,
    along with the radii sampled up to the specified maximum radius (must have units!)'''
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

def acquire_time_props(traj : mdt.Trajectory, time_points : np.ndarray, time_unit : Unit=nanosecond, properties : list[PolyProp]=DEFAULT_PROPS) -> pd.DataFrame:
    '''Compute and plot a battery of labelled and unit-ed properties over a given trajectory'''
    out_dframe = pd.DataFrame()

    if hasunits(time_points):
        time_points = time_points.in_units_of(time_unit) # convert to chosen time units - NOTE : must be done BEFORE extracting unit string
        unit_str = f' ({time_points.unit.get_symbol()})'
    else:
        unit_str = ''
    out_dframe[f'Sample Time{unit_str}'] = time_points

    for prop in properties:
        time_series = prop.compute(traj) 
        if (time_series.ndim > 1):
            time_series = time_series.sum(axis=-1) # sum over atoms / residue if multiple series are given (particular to SASA calculation)

        out_dframe[prop.label] = time_series * prop.unit

    return out_dframe

# Data formatting (for plotting)
def _dframe_splitter_factory(regex : str):
    '''Factory for creating splitting methods that choose an x-data column and remianinig y-data columns from a dataframe
    based on a regular expression applied to the name of the cdata columns'''
    def dframe_to_plot_data(dframe : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Takes a DataFrame populated with RDF data and unpacks it into x and y data and labels for plotting and calculations'''
        x_data = dframe.filter(regex=regex)
        y_data = dframe[[label for label in dframe.columns if label != x_data.columns[0]]] # index props with all non-time point columns

        return x_data, y_data
    return dframe_to_plot_data

rdfs_to_plot_data  = _dframe_splitter_factory(regex='Radius')
props_to_plot_data = _dframe_splitter_factory(regex='Time')