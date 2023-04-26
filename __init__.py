from . import analysis, charging, graphics, molutils, solvation
from . import extratypes, filetree, general, logutils, representation, simulation

LOGGERS_MASTER = [ # keep registry of all submodule loggers for ease of reference
    charging.LOGGER,
    representation.LOGGER,
    simulation.LOGGER
]