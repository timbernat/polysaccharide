from dataclasses import dataclass, field

# Useful type aliases
AtomIDMap = dict[str, dict[int, tuple[int, str]]]
ChargeMap = dict[int, float] 
ResidueChargeMap = dict[str, ChargeMap]

# Dataclasses for encapsulation
@dataclass
class Accumulator:
    '''Compact container for accumulating averages'''
    sum : float = 0.0
    count : int = 0

    @property
    def average(self) -> float:
        return self.sum / self.count