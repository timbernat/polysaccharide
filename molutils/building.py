import re

from rdkit import Chem
import mbuild as mb
from mbuild import Compound
from mbuild.lib.recipes.polymer import Polymer


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
        mb_comp, port_ids = mbmol_from_mono_smarts(mono_smarts)
        
        if re.search('TERM', res_name, flags=re.IGNORECASE): # check for "TERM" substring to identify terminal monomers
            assert(len(port_ids) == 1) # true terminal groups will only have a single port
            chain.add_end_groups(compound=mb_comp, index=port_ids[0], duplicate=False)
        else:
            chain.add_monomer(compound=mb_comp, indices=port_ids)

    chain.build(DOP) # assemble chain to desired size
    for atom in chain.particles():
        atom.charge = 0.0 # initialize all atoms as being uncharged (gets risk of pesky blocks of warnings)

    return chain