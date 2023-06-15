'''API for constructing modular sequential workflows for molecule building, charging, simulation, analysis, and more'''

import pkgutil, importlib
import logging

LOGGER = logging.getLogger(__name__)
for _loader, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'{__package__}.{_module_name}')
    globals()[_module_name] = module # register module to namespace
