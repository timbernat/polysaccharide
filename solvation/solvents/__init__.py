import importlib
import pkgutil
from typing import Any, Protocol


class SolventModule(Protocol):
    '''Represents a solvent module interface. Requires that each module implement a "register" method'''
    @staticmethod
    def register_to(registry : dict[str, Any]) -> None:
        pass


_solvent_plugins : list[SolventModule] = [
    # importlib.import_module(module_name, package=__file__)
    loader.find_module(module_name).load_module(module_name)
        for loader, module_name, ispkg in pkgutil.walk_packages(__path__)
            if ispkg # allows for standalone .py files to be placed parallel (but not below) this module
]

for _discovered_plugin in _solvent_plugins:
    _discovered_plugin.register_to(globals()) # add the registered name and solvent object to the current namespace
