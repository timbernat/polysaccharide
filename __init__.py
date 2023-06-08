'''Universal library for polymer representation parameterization, and simulation in OpenFF'''

from openff.toolkit import GLOBAL_TOOLKIT_REGISTRY as GTR
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper

from pathlib import Path
import openforcefields

import pkgutil, importlib

# Create molecule charging toolkit registries
GTR.register_toolkit(EspalomaChargeToolkitWrapper)
TOOLKITS = { 
    tk.toolkit_name : tk
        for tk in GTR.registered_toolkits
}

# Locate path where OpenFF forcefields are installed
OPENFF_DIR = Path(openforcefields.get_forcefield_dirs_paths()[0])

# Import submodules, register submodule loggers
LOGGERS_MASTER = [] # contains module-specific loggers for all submodules
for _loader, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'{__package__}.{_module_name}')
    globals()[_module_name] = module # register module to namespace

    if hasattr(module, 'LOGGER'): # make record of logger if one is present
        LOGGERS_MASTER.append(getattr(module, 'LOGGER'))