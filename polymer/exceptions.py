'''Custom Exceptions specific to Polymers and related objects'''

class SubstructMatchFailedError(Exception):
    '''Raised when molecule graph isomorphism match does not form a cover'''
    pass

class InsufficientChainLengthError(Exception):
    '''Raised when the polymer molecule being built is too short'''
    pass

class ExcessiveChainLengthError(Exception):
    '''Raised when the polymer molecule being built is too long'''
    pass

class CrosslinkingError(Exception):
    '''Raised when a polymer is crosslinked in a situation where it shouldn't be (or vice versa)'''
    pass

class AlreadySolvatedError(Exception):
    '''Raised when attempting to add solvent to a molecule which already has solvent'''
    pass

class ChargeMismatchError(Exception):
    '''Raised when attempting to merge two objects which disagree on their charging status'''
    pass

class NoSimulationsFoundError(Exception):
    '''Raised when attempting to load a simulation for a managed molecule when none are present'''
    pass

class MissingStructureData(Exception):
    '''Raised when a managed molecule has no associated structure file (e.g. PDB, SDF, etc.)'''
    pass

class MissingForceFieldData(Exception):
    '''Raised when a forcefield is unspecified for a Simulation or Interchange'''
    pass

class MissingMonomerData(Exception):
    '''Raised when no monomer information is found for a Polymer'''
    pass

class MissingMonomerDataUncharged(MissingMonomerData):
    '''Raised when no monomer information WITHOUT library charges is found for a Polymer'''
    pass

class MissingMonomerDataCharged(MissingMonomerData):
    '''Raised when no monomer information WITH library charges is found for a Polymer'''
    pass
