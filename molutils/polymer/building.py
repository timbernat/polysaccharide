'''Utilities for building new polymer structures; currently limited to linear polymers and PDB save format'''

# molecule building
import mbuild as mb
from mbuild import Compound
from mbuild.lib.recipes.polymer import Polymer as MBPolymer

# Monomer information
from . import abmono

# Typing and subclassing
from ..rdmol.rdtypes import *
SmartsByResidue = dict[str, str] # monomer SMARTS strings keyed by residue name

# Custom Exceptions for more tailored error messages
class SubstructMatchFailedError(Exception):
    pass

class InsufficientChainLengthError(Exception):
    pass


def mbmol_from_mono_smarts(SMARTS : str) -> tuple[Compound, list[int]]:
    '''Accepts a monomer-spec-compliant SMARTS string and returns an mbuild Compound and a list of the indices of hydrogen ports'''
    orig_rdmol = Chem.MolFromSmarts(SMARTS)

    orig_port_ids = []
    for atom in orig_rdmol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1) # convert wild-type atoms to hydrogens
            orig_port_ids.append(atom.GetIdx()) # record the positions of the ports in the original molecule

    mono_smiles = Chem.MolToSmiles(orig_rdmol)
    mb_compound = mb.load(mono_smiles, smiles=True)

    mb_ordered_rdmol = Chem.MolFromSmiles(mb_compound.to_smiles()) # create another molecule which has the same atom ordering as the mbuild Compound
    mb_ordered_rdmol = Chem.AddHs(mb_ordered_rdmol) # mbuild molecules don't have explicit Hs when converting to SMILES (although luckily AddHs adds them in the same order)
    mb_isomorphism = mb_ordered_rdmol.GetSubstructMatch(orig_rdmol) # determine mapping between original and mbuild atom indices
    if not mb_isomorphism: # ensure that the structures were in fact able to be matched before attempting backref map
        raise SubstructMatchFailedError
    mb_port_ids = [mb_isomorphism[idx] for idx in orig_port_ids]  # find the indices of the ports in the mbuild molecule

    return mb_compound, mb_port_ids

def build_linear_polymer(monomer_smarts : dict[str, str], DOP : int, add_Hs : bool=False, reverse_term_labels : bool=False) -> MBPolymer:
    '''Accepts a dict of monomer residue names and SMARTS (as one might find in a monomer JSON)
    and a degree of polymerization (i.e. chain length in number of monomers)) and returns an mbuild Polymer object'''
    chain = MBPolymer() 
    term_labels = ['head', 'tail'] # mbuild requires distinct labels in order to include both term groups
    if reverse_term_labels:
        term_labels = term_labels[::-1]

    for (resname, SMARTS) in monomer_smarts.items():
        mb_monomer, port_ids = mbmol_from_mono_smarts(SMARTS)
        
        if abmono.is_term_by_smarts(SMARTS):
            chain.add_end_groups(compound=mb_monomer, index=port_ids[0], label=term_labels.pop(), duplicate=False)
        else:
            chain.add_monomer(compound=mb_monomer, indices=port_ids)

    chain.build(DOP - 2, add_hydrogens=add_Hs) # "-2" is to account for term groups (in mbuild, "n" is the number of times to replicate just the middle monomers)
    for atom in chain.particles():
        atom.charge = 0.0 # initialize all atoms as being uncharged (gets risk of pesky blocks of warnings)

    return chain

def build_linear_polymer_limited(monomer_smarts : SmartsByResidue, max_chain_len : int, **build_args):
    '''Build a linear polymer which is no longer than the specified chain length'''
    DOP = abmono.estimate_max_DOP(monomer_smarts, max_chain_len=max_chain_len) # will raise error if length is unsatisfiable
    return build_linear_polymer(monomer_smarts, DOP=DOP, **build_args)