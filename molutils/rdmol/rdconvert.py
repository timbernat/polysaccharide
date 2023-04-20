from abc import ABC, abstractmethod, abstractproperty

from rdkit import Chem
from .rdtypes import *


class RDConverter(ABC):
    '''For converting an existing RDKit Molecule to and from a particular format to gain new properties'''
    @classmethod
    @abstractproperty
    def TAG(cls):
        pass

    @abstractmethod
    def convert(self, rdmol : RDMol) -> RDMol:
        pass

class SMARTSConverter(RDConverter):
    TAG = 'SMARTS'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmarts(Chem.MolToSmarts(rdmol))

class SMILESConverter(RDConverter):
    TAG = 'SMILES'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmiles(Chem.MolToSmiles(rdmol), sanitize=False)
    
class CXSMARTSConverter(RDConverter):
    '''Similar to SMARTSConverter but preserves the 3D structure'''
    TAG = 'CXSMARTS'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmarts(Chem.MolToCXSmarts(rdmol))

class CXSMILESConverter(RDConverter):
    '''Similar to SMILESConverter but preserves the 3D structure'''
    TAG = 'CXSMILES'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.MolFromSmiles(Chem.MolToCXSmiles(rdmol), sanitize=False)

class InChIConverter(RDConverter): # TOSELF : this does not preserve atom map num ordering (how to incorporate AuxInfo?)
    TAG = 'InChI'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.AddHs(Chem.MolFromInchi(Chem.MolToInchi(rdmol), removeHs=False, sanitize=False))
    
class JSONConverter(RDConverter):
    TAG = 'JSON'
    def convert(self, rdmol : RDMol) -> RDMol:
        return Chem.rdMolInterchange.JSONToMols(Chem.MolToJSON(rdmol))[0]


# define registry for convenient lookup once all child classes are defined above
RDCONVERTER_REGISTRY = { # keep easily accessible record of all available converter types
    child.TAG : child()
        for child in RDConverter.__subclasses__()
}