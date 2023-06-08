'''Custom Exceptions specific to Polymers and related objects'''

class SubstructMatchFailedError(Exception):
    pass

class InsufficientChainLengthError(Exception):
    pass

class ExcessiveChainLengthError(Exception):
    pass

class AlreadySolvatedError(Exception):
    pass

class ChargeMismatchError(Exception):
    pass

class MissingStructureData(Exception):
    pass

class MissingForceFieldData(Exception):
    pass

class MissingMonomerData(Exception):
    pass

class MissingMonomerDataUncharged(MissingMonomerData):
    pass

class MissingMonomerDataCharged(MissingMonomerData):
    pass
