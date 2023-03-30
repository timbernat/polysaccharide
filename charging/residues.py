from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from networkx import Graph
from rdkit.Chem.rdchem import Mol as RDMol
from .types import ChargeMap


# Interface for distributing non-integral monomer charges
class ChargeDistributionStrategy(ABC):
    '''Interface for defining how excess charge should be distributed within averaged residues
    to ensure an overall net 0 charge for each monomer fragment'''
    @abstractmethod
    def determine_distribution(self, net_charge : float, base_charges : ChargeMap, struct : Graph) -> ChargeMap:
        pass

class UniformDistributionStrategy(ChargeDistributionStrategy):
    '''Simplest possible strategy, distribute any excess charge evenly among all molecules in residue
    Each charge effectively becomes an average of averages when viewed in the context of the whole polymer'''
    def determine_distribution(self, net_charge : float, base_charges: ChargeMap, struct: Graph) -> ChargeMap:
        charge_offset = net_charge / len(base_charges) # net charge divided evenly amongst atoms (average of averages, effectively)
        return {sub_id : charge_offset for sub_id in base_charges}
    
# Representation class for residue charge averaging
@dataclass
class ChargedResidue:
    '''Dataclass for more conveniently storing averaged charges for a residue group'''
    charges : ChargeMap
    residue_name : str
    SMARTS : str
    mol_fragment : RDMol

    CDS : ChargeDistributionStrategy = field(default_factory=UniformDistributionStrategy) # set default strategy here

    def distrib_mono_charges(self) -> None:
        '''Distribute any excess charge amongst residue to ensure neutral, integral net charge'''
        net_charge = sum(chg for chg in self.charges.values())
        distrib = self.CDS.determine_distribution(net_charge, base_charges=self.charges, struct=self.mol_fragment)
        for sub_id, charge in self.charges.items():
            self.charges[sub_id] = charge - distrib[sub_id] # subtract respective charge offsets from each atom's partial charge