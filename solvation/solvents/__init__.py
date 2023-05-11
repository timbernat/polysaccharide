import importlib, pkgutil
from typing import Any, Protocol


class SolventModule(Protocol):
    '''Represents a solvent module interface. Requires that each module implement a "register" method'''
    @staticmethod
    def register_to(registry : dict[str, Any]) -> None:
        pass

_solvent_plugins : list[SolventModule] = [
    importlib.import_module(f'{__package__}.{module_name}')
    # loader.find_spec(module_name).loader.load_module(module_name)
        for loader, module_name, ispkg in pkgutil.iter_modules(__path__)
            if ispkg # allows for standalone .py files to be placed parallel (but not below) this module
]

for _discovered_plugin in _solvent_plugins:
    _discovered_plugin.register_to(globals()) # add the registered name and solvent object to the current namespace
