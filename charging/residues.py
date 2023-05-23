from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from networkx import Graph
from rdkit.Chem.rdchem import Mol as RDMol
from ..extratypes import ChargeMap


# Interface for distributing non-integral monomer charges
class ChargeDistributionStrategy(ABC):
    '''Interface for defining how excess charge should be distributed within averaged residues
    to ensure an overall net 0 charge for each monomer fragment'''
    @abstractmethod
    def determine_charge_offsets(self, base_charges : ChargeMap, fragment : Graph, charge_delta : float) -> ChargeMap:
        '''Provided a set of base charges, a molecule fragment for structure, and a desired charge in net charge, determine
        what charges offsets need to be applied where in order to achieve the desired net charge'''
        pass

    def redistributed_charges(self, base_charges : ChargeMap, fragment : Graph, desired_net_charge : float=0.0) -> ChargeMap:
        '''Take a map of base charges and a structural fragment for a residue and a desired net charge (typically neutral, i.e. 0)
        and return a new charge map with the excess/deficit charge distributed in such a way as to make the residue have the desired net charge'''
        charge_delta = desired_net_charge - sum(chg for chg in base_charges.values())
        charge_offsets = self.determine_charge_offsets(base_charges, fragment, charge_delta)

        new_charges = {
            sub_id : charge + charge_offsets[sub_id]
                for sub_id, charge in base_charges.items()
        }

        # assert(sum(chg for chg in new_charges.values()) == desired_net_charge) # double check charges were correctly redistributed - TODO : find more reliable way to check this than floating-point comparison
        return new_charges
            
class UniformDistributionStrategy(ChargeDistributionStrategy):
    '''Simplest possible strategy, distribute any excess charge in a residue according to a uniform distribution (spread evenly)'''
    def determine_charge_offsets(self, base_charges : ChargeMap, fragment : Graph, charge_delta : float) -> ChargeMap:
        return {sub_id : charge_delta / len(base_charges) for sub_id in base_charges}
    
# Representation class for residue charge averaging
@dataclass
class ChargedResidue:
    '''Dataclass for more conveniently storing averaged charges for a residue group'''
    charges : ChargeMap
    residue_name : str
    SMARTS : str
    mol_fragment : RDMol

    CDS : ChargeDistributionStrategy = field(default_factory=UniformDistributionStrategy) # set default strategy here

    def distrib_mono_charges(self, desired_net_charge : float=0.0) -> None:
        '''Distribute any excess charge amongst residue to ensure neutral, integral net charge'''
        self.charges = self.CDS.redistributed_charges(base_charges=self.charges, fragment=self.mol_fragment, desired_net_charge=desired_net_charge)