'''For defining sequential sets of molecule-related tasks (either serial or parallel)'''

# Typing and subclassing
from typing import Iterable, Optional, Union

# Custom Imports
from .components import WorkflowComponent
from ..general import asiterable

class Process:
    '''For defining control flow, logging, and execution of sets of Component tasks'''
    def __init__(self, components : Optional[Union[WorkflowComponent, Iterable[WorkflowComponent]]]=None) -> None:
        self.components = asiterable(components)