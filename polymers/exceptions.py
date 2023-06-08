'''Custom Exceptions specific to Polymers and related objects'''

class AlreadySolvatedError(Exception):
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
