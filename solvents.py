from pathlib import Path
from dataclasses import dataclass, field
from openmm.unit import gram, mole, centimeter

N_AVOGADRO = 6.02214076e23 * mole**-1


@dataclass
class Solvent:
    '''For encapsulating information about solvents'''
    name    : str
    formula : str
    smarts  : str

    density : float
    MW : float # molecular weight

    charges : dict[int, float] = field(default=None)
    structure_file : Path      = field(default=None)

    @property
    def number_density(self) -> float:
        '''
        Determine the number of solvent molecules per unit volume from known physical constants
        For best results, provide arguments as Quantities with associated units
        '''
        return (self.density / self.MW) * N_AVOGADRO

    @property
    def monomer_json_data(self):
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


# Define common solvents and their properties here
WATER_TIP3P = Solvent(
    name='water',
    formula = 'H2O',
    smarts  = '[#1:1]-[#8:3]-[#1:2]',
    density = 0.997 * (gram / centimeter**3), # at 300 K
    MW      = 18.015 * (gram / mole),
    charges = {
        "1" : 0.417, 
        "2" : 0.417,
        "3" : -0.834 
    },
)