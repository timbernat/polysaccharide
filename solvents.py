from .solvation import Solvent
from openmm.unit import gram, centimeter, mole

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