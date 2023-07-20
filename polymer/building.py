'''Utilities for building new polymer structures; currently limited to linear polymers and PDB save format'''

# logging
import logging
LOGGER = logging.getLogger(__name__)

# Molecule building
import mbuild as mb
from mbuild import Compound
from mbuild.lib.recipes.polymer import Polymer as MBPolymer

# Monomer information
from rdkit import Chem

from . import monomer
from ..molutils.rdmol import rdbond
from .monomer import is_linear, is_linear_homopolymer

# Typing and subclassing
from .exceptions import SubstructMatchFailedError, CrosslinkingError
from ..extratypes import ResidueSmarts


def mbmol_from_mono_smarts(SMARTS : str) -> tuple[Compound, list[int]]:
    '''Accepts a monomer-spec-compliant SMARTS string and returns an mbuild Compound and a list of the indices of hydrogen ports'''
    orig_rdmol = Chem.MolFromSmarts(SMARTS)
    orig_port_ids = rdbond.get_port_ids(orig_rdmol) # record indices of ports
    rdbond.hydrogenate_rdmol_ports(orig_rdmol, in_place=True) # replace ports with Hs to give complete fragments
    mono_smiles = Chem.MolToSmiles(orig_rdmol) # NOTE : CRITICAL that this be done AFTER hydrogenation (to avoid having ports in SMILES, which mbuild doesn't know how to handle)
    
    mb_compound = mb.load(mono_smiles, smiles=True)
    mb_ordered_rdmol = Chem.MolFromSmiles(mb_compound.to_smiles()) # create another molecule which has the same atom ordering as the mbuild Compound
    mb_ordered_rdmol = Chem.AddHs(mb_ordered_rdmol) # mbuild molecules don't have explicit Hs when converting to SMILES (although luckily AddHs adds them in the same order)
    Chem.Kekulize(mb_ordered_rdmol, clearAromaticFlags=True) # need to kekulize in order for aromatic bonds to be properly substructure matched (otherwise, ringed molecules are unsupported)

    mb_isomorphism = mb_ordered_rdmol.GetSubstructMatch(orig_rdmol) # determine mapping between original and mbuild atom indices
    if not mb_isomorphism: # ensure that the structures were in fact able to be matched before attempting backref map
        raise SubstructMatchFailedError
    mb_port_ids = [mb_isomorphism[idx] for idx in orig_port_ids]  # find the indices of the ports in the mbuild molecule

    return mb_compound, mb_port_ids

def build_linear_polymer(monomer_smarts : ResidueSmarts, DOP : int, add_Hs : bool=False, reverse_term_labels : bool=False) -> MBPolymer:
    '''Accepts a dict of monomer residue names and SMARTS (as one might find in a monomer JSON)
    and a degree of polymerization (i.e. chain length in number of monomers)) and returns an mbuild Polymer object'''
    if not is_linear(monomer_smarts):
        raise CrosslinkingError('Linear polymer building does not support non-linear monomer input')

    chain = MBPolymer() 
    term_labels = ['head', 'tail'] # mbuild requires distinct labels in order to include both term groups
    if reverse_term_labels:
        term_labels = term_labels[::-1]

    for (resname, SMARTS) in monomer_smarts.items():
        mb_monomer, port_ids = mbmol_from_mono_smarts(SMARTS)
        
        if monomer.is_term_by_smarts(SMARTS):
            chain.add_end_groups(compound=mb_monomer, index=port_ids[0], label=term_labels.pop(), duplicate=False)
        else:
            chain.add_monomer(compound=mb_monomer, indices=port_ids)

    LOGGER.info(f'Building linear polymer chain with {DOP} monomers ({monomer.estimate_chain_len(monomer_smarts, DOP)} atoms)')
    chain.build(DOP - 2, add_hydrogens=add_Hs) # "-2" is to account for term groups (in mbuild, "n" is the number of times to replicate just the middle monomers)
    for atom in chain.particles():
        atom.charge = 0.0 # initialize all atoms as being uncharged (gets risk of pesky blocks of warnings)

    return chain

def build_linear_polymer_limited(monomer_smarts : ResidueSmarts, max_chain_len : int, **build_args):
    '''Build a linear polymer which is no longer than the specified chain length'''
    DOP = monomer.estimate_DOP_lower(monomer_smarts, max_chain_len=max_chain_len) # will raise error if length is unsatisfiable
    return build_linear_polymer(monomer_smarts, DOP=DOP, **build_args)