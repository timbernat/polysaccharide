'''For recording, storing, and organizing parameters and files attached to a simulation'''

# General
import re
import numpy as np
from pathlib import Path

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# Typing and subclassing
from dataclasses import dataclass, field
from typing import Any

from ..filetree import JSONSerializable, JSONifiable
from .. import OPENFF_DIR

# Units, quantities, and dimensions
import openmm.unit
from openmm.unit import Quantity
from openmm.unit import picosecond, femtosecond
from openmm.unit import atmosphere, kelvin


@dataclass
class SimulationParameters(JSONifiable):
    '''For recording the parameters used to run an OpenMM Simulation'''
    total_time  : Quantity
    num_samples : int
    charge_method : str

    ensemble : str
    periodic : bool = True
    forcefield_name : str = ''

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

    @property
    def forcefield_path(self) -> Path:
        '''Returns the path to the official OpenFF Forcefield named in the parameter set'''
        ff_path = OPENFF_DIR / self.forcefield_name
        assert(ff_path.exists()) # make sure the forcefield requested genuinely exists
        
        return ff_path

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