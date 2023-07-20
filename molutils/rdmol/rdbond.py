'''For handling bonding ports and bond order up- and down-conversion'''

from .rdtypes import RDMol, RDAtom
from . import rdlabels
from ...general import optional_in_place, int_complement


# PORT FUNCTIONS
def is_port(atom : RDAtom) -> bool:
    '''Shorthand for deciding if an atom is a bonding site ("port")''' 
    return atom.GetAtomicNum() == 0

def get_port_ids(rdmol : RDMol) -> list[int]:
    '''Get atom indices of port (i.e. wild *-type or undefined) atoms'''
    return [
        atom.GetIdx()
            for atom in rdmol.GetAtoms()
                if is_port(atom)
    ]

neighbor_ports     = rdlabels._neighbor_factory_by_condition(    condition=is_port)
has_neighbor_ports = rdlabels._has_neighbor_factory_by_condition(condition=is_port)

@optional_in_place
def hydrogenate_rdmol_ports(rdmol : RDMol) -> None:
    '''Replace all port atoms with hydrogens'''
    for port_id in get_port_ids(rdmol):
        rdmol.GetAtomWithIdx(port_id).SetAtomicNum(1)


# BONDING FUNCTIONS
from ...general import optional_in_place
from .rdtypes import *

## bond order prechecks
class BondOrderModificationError(Exception):
    pass

def bond_order_increasable(rdmol : RDMol, *atom_pair_ids : list[int, int]) -> bool:
    '''Check if both atoms have a free neighboring port'''
    return all(
        has_neighbor_ports(rdmol.GetAtomWithIdx(atom_id))
            for atom_id in atom_pair_ids
    )

def are_bonded_atoms(rdmol : RDMol, *atom_pair_ids : list[int, int]) -> bool:
    '''Check if pair of atoms in an RDMol have a bond between then'''
    return (rdmol.GetBondBetweenAtoms(*atom_pair_ids) is not None)
bond_order_decreasable = are_bonded_atoms # alias for cohesiveness (can't decrease bond order if no bond exists)

# bond order modifiers
@optional_in_place
def increase_bond_order(rwmol : RWMol, *bond_atom_ids : list[int, int], prioritize_unlabelled_ports : bool=True) -> None:
    '''Exchange two ports for a bond of one higher order in a modifiable RWMol'''
    if not bond_order_increasable(rwmol, *bond_atom_ids):
        raise BondOrderModificationError

    # determine expected bond type after order increase (handle single-bond removal, specifically) 
    curr_bond = rwmol.GetBondBetweenAtoms(*bond_atom_ids)
    if curr_bond is None:
        new_bond_type = Chem.BondType.SINGLE # with no pre-existing bond, simply add a single bond
    else: 
        new_bond_type = Chem.BondType.values[curr_bond.GetBondTypeAsDouble() + 1] # with pre-existing bond, need to get the next order up by numeric lookup
        rwmol.RemoveBond(*bond_atom_ids) # also remove the existing bond for new bond creation

    # create new bond
    rwmol.AddBond(*bond_atom_ids, order=new_bond_type) # create new bond or specified order
    
    # remove ports on newly-bonded atoms
    for atom_id in bond_atom_ids: 
        atom = rwmol.GetAtomWithIdx(atom_id)
        nb_ports = neighbor_ports(atom)
        if prioritize_unlabelled_ports:
            nb_ports = iter(sorted(nb_ports, key=lambda port : bool(port.GetAtomMapNum()))) # sort with unlabelled ports first - make into iter to permit next() call

        nb_port = next(nb_ports) # guaranteed not to raise StopIteration by the bond_order_increasable check at the start
        rwmol.RemoveAtom(nb_port.GetIdx())

@optional_in_place
def decrease_bond_order(rwmol : RWMol, *bond_atom_ids : list[int, int], dummyLabels : bool=True) -> RWMol: 
    '''Exchange a bond for two ports and a bond of lower order'''
    if not bond_order_decreasable(rwmol, *bond_atom_ids):
        raise BondOrderModificationError
    
    # determine expected bond type after order decrease (handle single-bond case, specifically) 
    curr_bond = rwmol.GetBondBetweenAtoms(*bond_atom_ids) # guaranteed to not be None by the bond_order_decreasable check at the start
    new_bond_type = Chem.BondType.values[curr_bond.GetBondTypeAsDouble() - 1] # with pre-existing bond, need to get the next order up by numeric lookup
    if new_bond_type == Chem.BondType.UNSPECIFIED:
        new_bond_type = None # explicitly set to NoneType if single bond is broken

    # remove existing bond; not single bond, replace with bond of new type
    rwmol.RemoveBond(*bond_atom_ids) # create new bond or specified order
    if new_bond_type is not None:
        rwmol.AddBond(*bond_atom_ids, order=new_bond_type)
    
    # add new ports for broken bond
    free_isotope_labels = int_complement(rdlabels.get_isotopes(rwmol, unique=True), bounded=False) # generate unused isotope labels
    for atom_id in bond_atom_ids:
        new_port = RDAtom('*') 
        if dummyLabels: # label with unique isotope number (a la FragmentOnBonds) if specified
            new_port.SetIsotope(next(free_isotope_labels))
        
        new_port_id = rwmol.AddAtom(new_port)# insert new port into molecule, taking note of index (TOSELF : ensure that this inserts indices at END of existing ones, could cause unexpected modification if not)
        rwmol.AddBond(atom_id, new_port_id) # bond the atom to the new port