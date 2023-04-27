'''For calculating information about a polymer chain ab mono (i.e. from monomer information)'''

# Numeric and string processing
import re
import numpy as np

# Typing and subclassing
from ..rdmol.rdtypes import *
SmartsByResidue = dict[str, str] # monomer SMARTS strings keyed by residue name

# Custom Exceptions for more tailored error messages
class InsufficientChainLengthError(Exception):
    pass


# Estimating chain parameters from monomer info
def get_num_ports(rdmol : RDMol) -> int:
    '''Counts the number of ports present in a monomer SMARTS'''
    # return Chem.MolToSmarts(SMARTS).count('*') # naive but relatively effective for nominal cases
    return sum(1 for atom in rdmol.GetAtoms() if not atom.GetAtomicNum())

is_term_by_resname = lambda res_name : bool(re.search('TERM', res_name, flags=re.IGNORECASE)) # naive and far less general test when explicitly labelled
is_term_by_smarts  = lambda SMARTS : get_num_ports(Chem.MolFromSmarts(SMARTS)) == 1 # terminal monomers must have exactly 1 port by definition
is_term_by_rdmol   = lambda rdmol : get_num_ports(rdmol) == 1

def count_middle_and_term_mono(monomer_smarts : SmartsByResidue) -> tuple[int, int]:
    '''Determine how many of the monomers in a base set are middle vs terminal
    Results return is number of middle monomers, followed by the number of terminal monomers'''
    group_counts = [0, 0]
    for mono_SMARTS in monomer_smarts.values():
        group_counts[is_term_by_smarts(mono_SMARTS)] += 1 # index by bool
    
    n_mid, n_term = group_counts # unpack purely for documentation and statification
    return (n_mid, n_term)

def is_linear_polymer(monomer_smarts : SmartsByResidue) -> bool:
    '''Identify if a polymer is a linear, unbranched chain'''
    n_mid, n_term = count_middle_and_term_mono(monomer_smarts)
    return (n_term == 2)

def is_homopolymer(monomer_smarts : SmartsByResidue) -> bool:
    '''Identify if a polymer is a homopolymer (i.e. only 1 type of middle monomer)'''
    n_mid, n_term = count_middle_and_term_mono(monomer_smarts)
    return (n_mid == 1)

def is_linear_homopolymer(monomer_smarts : SmartsByResidue) -> bool:
    '''Identify if a polymer is a linear homopolymer'''
    return is_linear_polymer(monomer_smarts) and is_homopolymer(monomer_smarts)

def estimate_chain_len(monomer_smarts : SmartsByResidue, DOP : int) -> int:
    '''Given a set of monomers and the desired degree of polymerization, estimate the length of the resulting chain
    !NOTE! : As-implemented, only works for linear homopolymers and block copolymers with equal an distribution of monomers'''
    num_mono = len(monomer_smarts)

    mono_term   = np.zeros(num_mono, dtype=bool) #  terminality of each monomer (i.e. whether or not it is a term group)
    mono_multip  = np.zeros(num_mono, dtype=int) # multiplicity of each polymer (i.e. how many times is occurs in a chain)
    mono_contrib = np.zeros(num_mono, dtype=int) # contribution of each monomer (i.e. how many atoms does it add to the chain)

    for i, (resname, SMARTS) in enumerate(monomer_smarts.items()):
        monomer = Chem.MolFromSmarts(SMARTS)

        num_atoms = monomer.GetNumAtoms()
        num_ports = get_num_ports(monomer)
        is_term = is_term_by_rdmol(monomer)

        mono_term[i] = is_term
        mono_multip[i] = is_term # temporarily set middle monomer contribution to 0
        mono_contrib[i] = num_atoms - num_ports

    num_term = sum(mono_term)
    num_mid  = num_mono - num_term # assumed that all monomers are either terminal or not
    mono_multip[~mono_term] = (DOP - num_term) / num_mid # naive assumption that all middle monomers contribute rest of chain equally (for homopolymers, this is always true)

    return mono_contrib @ mono_multip # compute dot product to yield final count

def estimate_max_DOP(monomer_smarts : SmartsByResidue, max_chain_len : int, min_DOP : int=3) -> int:
    '''Returns the largest DOP for a set of monomers which yields a chain no longer than the specified chain length'''
    base_chain_len = estimate_chain_len(monomer_smarts, min_DOP)
    if base_chain_len > max_chain_len: # pre-check when optimization is impossible
        raise InsufficientChainLengthError(f'Even shortest possible chain (DOP={min_DOP}, N={base_chain_len}) is longer than the specified max length of {max_chain_len} atoms')

    DOP = min_DOP 
    while estimate_chain_len(monomer_smarts, DOP + 1) < max_chain_len: # check if adding 1 more monomer keeps the length below the threshold
        DOP += 1

    return DOP