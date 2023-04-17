from . import properties, trajectory, propplot, calculation

import mdtraj as mdt
from openmm.unit import nanometer, dimensionless


DEFAULT_PROPS = [ # the base properties of interest for the 2023 monomer spec study
    properties.PolyProp(calc=mdt.compute_rg                , name='Radius of Gyration'             , abbr='Rg'  , unit=nanometer),
    properties.PolyProp(calc=mdt.shrake_rupley             , name='Solvent Accessible Surface Area', abbr='SASA', unit=nanometer**2, kwargs={'mode' : 'residue'}),
    properties.PolyProp(calc=mdt.relative_shape_antisotropy, name='Relative Shape Anisotropy'      , abbr='K2'  , unit=dimensionless)    
]