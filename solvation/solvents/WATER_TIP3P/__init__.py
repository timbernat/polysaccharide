# TOSELF : might be nice to find a way to may this relative for transferability (standard "." notation gives ImportError: attempted relative import beyond top-level package)
from polymer_utils.solvation.solvent import Solvent 
from openmm.unit import gram, centimeter, mole
from pathlib import Path


WATER_TIP3P = Solvent(
    name = 'water2',
    formula = 'H2O',
    smarts  = '[#1:1]-[#8:3]-[#1:2]',

    density = 0.997 * (gram / centimeter**3), # at 300 K
    MW      = 18.015 * (gram / mole),
    charges = {
        "1" : 0.417, 
        "2" : 0.417,
        "3" : -0.834 
    },
    structure_file  = Path(__file__).parent/'water.json',
    forcefield_file = Path(__file__).parent/'tip3p.offxml'
)

def register_to(registry : dict) -> None:
    '''Add solvent to list of registered plugins'''
    registry['WATER_TIP3P'] = WATER_TIP3P