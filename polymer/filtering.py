'''Utilities for filtering polymers'''

# Custom imports
from .representation import Polymer
from ..simulation.records import SimulationPaths, SimulationParameters

# Typing and Subclassing
from typing import Any, Callable, TypeAlias


# Filtering Polymers by attributes
MolFilter = Callable[[Polymer], bool] # typing template for explicitness

def filter_factory_by_attr(attr_name : str, condition : Callable[[Any], bool]=bool, inclusive : bool=True) -> MolFilter:
    '''Factory function for creating MolFilters for arbitrary individual Polymer attributes

    Takes the name of the attribute to look-up and a truthy "condition" function to validate attribute values (by default just bool() i.e. property exists)
    Resulting filter will return Polymers for which the condition is met by default, or those for which it isn't if inclusive == False
    '''
    def _filter(polymer : Polymer) -> bool:
        '''The actual case-specific filter '''
        attr_val = getattr(polymer,  attr_name)
        satisfied = condition(attr_val)
        
        if inclusive:
            return satisfied
        return not satisfied # invert truthiness if non-inclusive

    return _filter

identity          = lambda polymer : True # no filter, return all polymers (or conversely None if non-inclusive)
has_sims          = filter_factory_by_attr('completed_sims')
has_monomers      = filter_factory_by_attr('has_monomer_info')
has_monomers_chgd = filter_factory_by_attr('has_monomer_info_charged')

is_charged        = filter_factory_by_attr('charges')
is_uncharged      = filter_factory_by_attr('charges', inclusive=False)

is_solvated       = filter_factory_by_attr('solvent')
is_unsolvated     = filter_factory_by_attr('solvent', inclusive=False)

is_AM1_sized      = filter_factory_by_attr('n_atoms', condition=lambda n : 0 < n <= 300)
is_base           = lambda polymer : polymer.base_mol_name == polymer.mol_name
# has_monomers = lambda polymer : polymer.has_monomer_info 
# AM1_sized    = lambda polymer : 0 < polymer.n_atoms <= 300

# Filtering simulation directories
SimDirFilter : TypeAlias = Callable[[SimulationPaths, SimulationParameters], bool]
has_binary_traj = lambda sim_paths, sim_params : (sim_params.binary_traj == True)