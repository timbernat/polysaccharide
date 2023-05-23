from dataclasses import dataclass, field
from pathlib import Path

from ..extratypes import MonomerInfo
from openmm.unit import AVOGADRO_CONSTANT_NA


@dataclass(frozen=True, eq=True) # allows hashability and comparison for solvents
class Solvent:
    '''
    Representation class for encapsulating information about solvents

    Specific solvents are implemented as plugins in .solvent_plugins; must include:
    -- An __init__.py containing an instance of Solvent along with a register() method
    -- A .PDB structure file (which must also be specified in the definition on the above solvent)
    -- An .OFFXML force field file (which must also be specified in the definition on the above solvent)
    '''
    name    : str
    formula : str
    smarts  : str

    density : float # at a specified P and T
    MW : float      # molecular weight

    charges : dict[int, float]
    structure_file  : Path = field(compare=False) # allows for match when transferring files between machines (local structure path will be different)
    forcefield_file : Path = field(compare=False) # allows for match when transferring files between machines (local forcefield path will be different)

    def __hash__(self): # custom hash method needed as unit-ful attributes (such as density and MW) are unhashable Quantity objects
        return hash((self.name, self.formula, self.smarts))

    @property
    def number_density(self) -> float:
        '''
        Determine the number of solvent molecules per unit volume from known physical constants
        For best results, provide arguments as Quantities with associated units
        '''
        return (self.density / self.MW) * AVOGADRO_CONSTANT_NA

    @property # TOSELF : slated for deprecation once converted over to MonomerInfo representation
    def monomer_json_data(self): # TOSELF : given th plugin spec, could potentially just make this read a JSON?
        '''Generate a monomer-spec-conformant JSON dictionary entry'''
        return {
            "monomers": {
                self.name : self.smarts
            },
            "caps": {
                self.name : []
            },
            "charges" : {
                self.name : self.charges
            }
        }
    
    @property
    def monomer_info(self) -> MonomerInfo:
        '''Generate monomer information representation'''
        return MonomerInfo(
            monomers={
                self.name : self.smarts
            },
            charges={
                self.name : self.charges
            }
        )
