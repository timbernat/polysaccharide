'''For defining sequential sets of molecule-related tasks (either serial or parallel)'''

# logging
import logging

from .. import LOGGERS_MASTER
from ..logutils import ProcessLogHandler

# Generic imports
from pathlib import Path

# Typing and subclassing
from typing import Iterable, Optional, Union

# Custom Imports
from .components import WorkflowComponent
from .. import general

from ..polymer.representation import Polymer
from ..polymer.management import PolymerFunction


@general.generate_repr(lookup_attr='DISP_ATTRS')
class Process:
    '''For defining control flow, logging, and execution of sets of Component tasks'''
    
    # SETUP
    def __init__(self, components : Union[WorkflowComponent, Iterable[WorkflowComponent]]=None, proc_name : str='') -> None:
        self.components = [] # distinct from __init__'s components, stored running set of Components as Process is setup and modified
        if components is None:
            components = []
        self.add_components(components)

        self.proc_name = proc_name
        self.id = id(self)

        tag_elems = ['Process', str(self.id)]
        if proc_name:
            tag_elems.append(proc_name) 
        self.tag = '_'.join(tag_elems)

        self.logger = logging.getLogger(self.tag) # generate logger with unique ID key

    DISP_ATTRS = ( # attributes to display when defining boilerplate __repr__ method
        'proc_name',
        'id',
        'components'
    )

    def add_components(self, new_components : Iterable[WorkflowComponent]) -> None:
        '''Extend the Process with a series of newly-created component'''
        self.components.extend(general.asiterable(new_components))

    def add_component(self, new_component : WorkflowComponent) -> None:
        '''Add a single new component (singleton version of self.add_components)'''
        # self.add_components([new_component])
        self.components.append(new_component)

    # PROPERTIES
    @property
    def polymer_fns(self) -> Iterable[PolymerFunction]:
        '''Generate the sequence of PolymerFunctions for the current Components'''
        return [comp.make_polymer_fn() for comp in self.components]

    @property
    def collated_polymer_fn(self) -> PolymerFunction:
        '''Collate together the functions from all constituent components into a single unified polymer function'''
        def polymer_fn(polymer : Polymer, poly_logger : logging.Logger) -> None:
            '''Run OpenMM simulation(s) according to sets of predefined simulation parameters'''
            for sub_poly_fn in self.polymer_fns:
                sub_poly_fn(polymer, poly_logger)

        return polymer_fn
    
    # EXECUTION AND DISPATCH
    def execute(self, targ_mols : Iterable[Polymer], sequential : bool=True, log_output_dir : Path=Path.cwd(),
                loggers : Union[logging.Logger, list[logging.Logger]]=LOGGERS_MASTER) -> None:
        '''
        Execute actions specified by Components over all molecules in a Collection
        If sequential, each action in order will be executed over all molecules before moving to the next action
        If non-sequential, all actions will be performed for the first molecule, then the next, and so on
        '''
        loggers = [self.logger] + loggers # create copy of loggers with presonal logger injected

        proc_iter = general.swappable_loop_order(
            general.progress_iter(targ_mols, key=lambda poly : poly.mol_proc_name),
            general.progress_iter(self.components, key=lambda comp : comp.__class__.__proc_name__),
            swap=sequential # switch order of iteration in sequential mode
        )

        # if sequential:
        with ProcessLogHandler(filedir=log_output_dir, loggers=loggers, proc_name=self.proc_name, timestamp=True) as msf_handler:
            for (comp_str, comp), (poly_str, polymer) in proc_iter:
                self.logger.info(f'Executing Component {comp_str} on Molecule {poly_str}')
                poly_fn = comp.make_polymer_fn()

                with msf_handler.subhandler(filedir=polymer.logs, loggers=loggers, proc_name=self.proc_name, timestamp=True) as subhandler: # also log actions to individual Polymers
                    poly_fn(polymer, subhandler.personal_logger)
