'''Custom exceptions specific to polymers'''

class SubstructMatchFailedError(Exception):
    pass

class InsufficientChainLengthError(Exception):
    pass

class ExcessiveChainLengthError(Exception):
    pass