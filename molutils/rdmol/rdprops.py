'''For manipulating RDKit molecule and atom properties (Props)'''

# Typing and Subclassing
from typing import Any, Optional, Union
from .rdtypes import RDMol, RDAtom
from openff.toolkit.topology.molecule import Atom


# For lookup of varoius C++ type-specific methods RDKit enforces
RDPROP_GETTERS = {
    str   : 'GetProp',
    bool  : 'GetBoolProp',
    int   : 'GetIntProp',
    float : 'GetDoubleProp'
}
RDPROP_SETTERS = {
    str   : 'SetProp',
    bool  : 'SetBoolProp',
    int   : 'SetIntProp',
    float : 'SetDoubleProp'
}

# Property transferring functions
def copy_rd_props(from_rdobj : Union[RDAtom, RDMol], to_rdobj : Union[RDAtom, RDMol]) -> None:
    '''For copying properties between a pair of RDKit Atoms or Mols'''
    # NOTE : will avoid use of GetPropsAsDict() to avoid errors from restrictive C++ typing
    # for prop, prop_val in from_rdobj.GetPropsAsDict().items():
    #     to_rdobj.SetProp(prop, prop_val)
    for prop in from_rdobj.GetPropNames():
        to_rdobj.SetProp(prop, from_rdobj.GetProp(prop))

def copy_atom_metadata(offatom : Atom, rdatom : RDAtom, preserve_type : bool=True) -> None:
    '''Copies all attributes from the metadata dict of an OpenFF-type Atom as Props of an RDKit-type atom'''

    for key, value in offatom.metadata.items():
        if (type(value) not in RDPROP_SETTERS) or (not preserve_type): # set as string if type is unspecified or if explicitly requested to
            rdatom.SetProp(key, str(value))
        else:
            setter = getattr(rdatom, RDPROP_SETTERS[type(value)]) # use the atom's setter for the appropriate type
            setter(key, value)

# Property inspection functions
def aggregate_atom_prop(rdmol : RDMol, prop : str, prop_type : type=str) -> list[Any]:
    '''Collects the values of a given Prop across all atoms in an RDKit molecule'''
    getter_type = RDPROP_GETTERS[prop_type]
    return [
        getattr(atom, getter_type)(prop)
            for atom in rdmol.GetAtoms()
    ]

# Property modification functions - TODO : add non-inplace versions of all of these methods
def hydrogenate_rdmol_ports(rdmol : RDMol, return_port_ids : bool=False) -> Optional[list[int]]:
    '''Replace all ports (i.e. wild *-type atoms) in a molecule with hydrogens'''
    port_ids = []
    for atom in rdmol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1) # convert wild-type atoms to hydrogens
            port_ids.append(atom.GetIdx()) # record the positions of the ports in the original molecule

    if return_port_ids:
        return port_ids
    
def assign_ordered_atom_map_nums(rdmol : RDMol) -> None:
    '''Assigns atom's id to its atom map number for all atoms in an RDmol'''
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()) # need atom map numbers to preserve positional mapping in SMARTS
        # atom.SetAtomMapNum(atom.GetIdx() + 1) # need atom map numbers to preserve positional mapping in SMARTS; "+1" avoids mapping any atoms to 0

def clear_atom_map_nums(rdmol : RDMol) -> None:
    '''Removes atom map numbers from all atoms in an RDMol'''
    for atom in rdmol.GetAtoms():
        atom.SetAtomMapNum(0)