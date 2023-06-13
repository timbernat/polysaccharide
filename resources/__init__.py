from pathlib import Path
from typing import Iterable

import pkgutil, importlib
import importlib.resources as impres

import logging
LOGGER = logging.getLogger(__name__)

def non_dunder(dir : Path) -> Iterable[str]:
    '''Return all subpaths of a directory dir which contain no double underscores'''
    assert(dir.is_dir())
    return [
        path
            for path in dir.iterdir()
                if '__' not in path.name
    ]

RESOURCE_PATH = impres.files(__package__)

AVAIL_RESOURCES = {} # load submodules, record available path assets
for _loader, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'{__package__}.{_module_name}')
    globals()[_module_name] = module # register module to namespace
    AVAIL_RESOURCES[_module_name] = non_dunder(impres.files(module))

