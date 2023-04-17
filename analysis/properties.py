# Typing and Subclassing
from mdtraj import Trajectory
from typing import Any, Callable
from dataclasses import dataclass, field
from openmm.unit import Unit


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

    def compute(self, traj : Trajectory) -> Any:
        '''Apply method to a trajectory'''
        return self.calc(traj, **self.kwargs)
    
