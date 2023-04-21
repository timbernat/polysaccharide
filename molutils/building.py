# Numeric and string processing
import re
import numpy as np

# molecule building
import mbuild as mb
from mbuild import Compound
from mbuild.lib.recipes.polymer import Polymer

# Typing and subclassing
from polymer_utils.molutils.rdmol.rdtypes import *


# Estimating chain parameters prior to building from monomer info
def get_num_ports(rdmol : RDMol) -> int:
    '''Counts the number of ports present in a monomer SMARTS'''
    # return Chem.MolToSmarts(SMARTS).count('*') # naive but relatively effective for nominal cases
    return sum(1 for atom in rdmol.GetAtoms() if not atom.GetAtomicNum())

is_term_by_resname = lambda res_name : bool(re.search('TERM', res_name, flags=re.IGNORECASE)) # naive and far less general test when explicitly labelled
is_term_by_smarts  = lambda SMARTS : get_num_ports(Chem.MolFromSmarts(SMARTS)) == 1 # terminal monomers must have exactly 1 port by definition
is_term_by_rdmol   = lambda rdmol : get_num_ports(rdmol) == 1

def estimate_chain_len(monomer_structs : dict[str, str], DOP : int) -> int:
    '''Given a set of monomers and the desired degree of polymerization, estimate the length of the resulting chain
    NOTE : As-implemented, only works for linear homopolymers'''
    num_mono = len(monomer_structs)

    mono_term   = np.zeros(num_mono, dtype=bool) #  terminality of each monomer (i.e. whether or not it is a term group)
    mono_multip  = np.zeros(num_mono, dtype=int) # multiplicity of each polymer (i.e. how many times is occurs in a chain)
    mono_contrib = np.zeros(num_mono, dtype=int) # contribution of each monomer (i.e. how many atoms does it add to the chain)

    for i, (resname, SMARTS) in enumerate(monomer_structs.items()):
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

# mbuild assembly
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
    mb_port_ids = [mb_isomorphism[idx] for idx in orig_port_ids]  # find the indices of the ports in the mbuild molecule

    return mb_compound, mb_port_ids

def build_linear_polymer(monomer_structs : dict[str, str], DOP : int) -> Polymer:
    '''Accepts a dict of monomer residue names and SMARTS (as one might find in a monomer JSON)
    and a degree of polymerization (i.e. chain length in number of monomers)) and returns an mbuild Polymer object'''
    chain = Polymer() 
    for res_name, mono_smarts in monomer_structs.items():
        mb_monomer, port_ids = mbmol_from_mono_smarts(mono_smarts)
        
        if is_term_by_smarts(mono_smarts):
            chain.add_end_groups(compound=mb_monomer, index=port_ids[0], duplicate=False)
        else:
            chain.add_monomer(compound=mb_monomer, indices=port_ids)

    chain.build(DOP) # assemble chain to desired size
    for atom in chain.particles():
        atom.charge = 0.0 # initialize all atoms as being uncharged (gets risk of pesky blocks of warnings)

    return chain