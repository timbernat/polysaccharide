'''Simplifies creation of Simulations which correspond to a particular thermodynamic ensemble'''

# Typing and subclassing
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Iterable, Optional, Union

# Logging
import logging
LOGGER = logging.getLogger(__name__)

# OpenForceField
from openff.units import unit as offunit # need OpenFF version of unit for Interchange positions for some reason
from openff.interchange import Interchange

from openmm import Integrator, VerletIntegrator, LangevinMiddleIntegrator
from openmm.openmm import Force, MonteCarloBarostat
from openmm.app import Simulation

 # Custom Imports
from .preparation import create_simulation
from .records import SimulationParameters


class EnsembleSimulationFactory(ABC):
    '''Base class for implementing interface for generating ensemble-specific simulations'''
    # Abstract methods and properties
    @abstractproperty
    @classmethod
    def ensemble(self) -> str:
        '''Specify state variables of ensemble'''
        pass

    @abstractproperty
    @classmethod
    def ensemble_name(self) -> str:
        '''Specify name of ensemble'''
        pass

    @abstractmethod
    def integrator(self, sim_params : SimulationParameters) -> Integrator:
        '''Specify how to integrate forces in each timestep'''
        pass

    @abstractmethod
    def forces(self, sim_params : SimulationParameters) -> Optional[Iterable[Force]]:
        '''Specify any additional force contributions to position/velocity updates'''
        pass

    # Concrete methods and properties
    _REPR_ATTRS = ('ensemble', 'ensemble_name')
    def __repr__(self) -> str:
        '''Provide a description of the ensemble and mechanics used'''
        
        attr_str = ', '.join(
            f'{attr_name}={getattr(self, attr_name)}'    
                for attr_name in self._REPR_ATTRS
        )
        return f'{self.__class__.__name__}({attr_str})'
    
    @property
    def desc(self) -> str:
        '''Verbal description of ensemble'''
        return f'{self.ensemble} ({self.ensemble_name.capitalize()} ensemble)'

    def create_simulation(self, interchange : Interchange, sim_params : SimulationParameters) -> Simulation:
        '''Generate an OpenMM Simulation instance using the Forces and Integrator defined for the ensemble of choice'''
        integrator = self.integrator(sim_params)
        forces     = self.forces(sim_params)

        sim = create_simulation(
            interchange,
            integrator=integrator,
            forces=forces
        )

        desc_str = f'Created {self.desc} Simulation with {integrator.__class__.__name__}'
        if forces:
            force_str = ', '.join(force.__class__.__name__ for force in forces)
            desc_str = f'{desc_str} and {force_str} forces'
        LOGGER.info(desc_str)
        
        return sim

    @classmethod
    @property
    def registry(self) -> dict[str, 'EnsembleSimulationFactory']:
        '''Easily accessible record of all available concrete ensemble implementations'''
        return {  
            ens_factory.ensemble : ens_factory
                for ens_factory in EnsembleSimulationFactory.__subclasses__()
        }

# Concrete implementations
class NVESimulationFactory(EnsembleSimulationFactory):
    ensemble = 'NVE'
    ensemble_name = 'microcanonical'

    def integrator(self, sim_params: SimulationParameters) -> Integrator:
        return VerletIntegrator(sim_params.timestep)
    
    def forces(self, sim_params: SimulationParameters) -> Optional[Iterable[Force]]:
        return None
    
class NVTSimulationFactory(EnsembleSimulationFactory):
    ensemble = 'NVT'
    ensemble_name = 'canonical'

    def integrator(self, sim_params: SimulationParameters) -> Integrator:
        return LangevinMiddleIntegrator(sim_params.temperature, sim_params.friction_coeff, sim_params.timestep)
    
    def forces(self, sim_params: SimulationParameters) -> Optional[Iterable[Force]]:
        return None
    
class NPTSimulationFactory(EnsembleSimulationFactory):
    ensemble = 'NPT'
    ensemble_name = 'isothermal-isobaric'

    def integrator(self, sim_params: SimulationParameters) -> Integrator:
        return LangevinMiddleIntegrator(sim_params.temperature, sim_params.friction_coeff, sim_params.timestep)
    
    def forces(self, sim_params: SimulationParameters) -> Optional[Iterable[Force]]:
        return [MonteCarloBarostat(sim_params.pressure, sim_params.temperature, sim_params.barostat_freq)]
    
  