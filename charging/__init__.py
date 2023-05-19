'''Utilities for calculating, averaging, and applying partial charges to Molecules'''

import pkgutil, importlib
import logging

from openff.toolkit import GLOBAL_TOOLKIT_REGISTRY as GTR
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper

# Create molecule charging toolkit registries
GTR.register_toolkit(EspalomaChargeToolkitWrapper)
TOOLKITS = { 
    tk.toolkit_name : tk
        for tk in GTR.registered_toolkits
}

# import submodules, register logger
LOGGER = logging.getLogger(__name__)
for _loader, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'{__package__}.{_module_name}')
    globals()[_module_name] = module # register module to namespace