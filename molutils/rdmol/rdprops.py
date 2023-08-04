'''For manipulating RDKit molecule and atom properties (Props)'''

# Typing and Subclassing
from typing import Any, Union

from .rdtypes import RDMol, RDAtom
from openff.toolkit.topology.molecule import Atom as OFFAtom

# Custom imports
from ...general import optional_in_place

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

# PROPERTY TRANSFER FUNCTIONS
def copy_rd_props(from_rdobj : Union[RDAtom, RDMol], to_rdobj : Union[RDAtom, RDMol]) -> None:
    '''For copying properties between a pair of RDKit Atoms or Mols'''
    # NOTE : will avoid use of GetPropsAsDict() to avoid errors from restrictive C++ typing
    # for prop, prop_val in from_rdobj.GetPropsAsDict().items():
    #     to_rdobj.SetProp(prop, prop_val)
    for prop in from_rdobj.GetPropNames():
        to_rdobj.SetProp(prop, from_rdobj.GetProp(prop))

def copy_atom_metadata(offatom : OFFAtom, rdatom : RDAtom, preserve_type : bool=True) -> None:
    '''Copies all attributes from the metadata dict of an OpenFF-type Atom as Props of an RDKit-type atom'''
    for key, value in offatom.metadata.items():
        if (type(value) not in RDPROP_SETTERS) or (not preserve_type): # set as string if type is unspecified or if explicitly requested to
            rdatom.SetProp(key, str(value))
        else:
            setter = getattr(rdatom, RDPROP_SETTERS[type(value)]) # use the atom's setter for the appropriate type
            setter(key, value) # pass key and value to setter method

# PROPERTY INSPECTION FUNCTIONS
def atom_ids_with_prop(rdmol : RDMol, prop_name : str) -> list[int]:
    '''Returns list of atom IDs of atom which have a particular property assigned'''
    return [
        atom.GetIdx()
            for atom in rdmol.GetAtoms()
                if atom.HasProp(prop_name)
    ]

def aggregate_atom_prop(rdmol : RDMol, prop : str, prop_type : type=str) -> list[Any]:
    '''Collects the values of a given Prop across all atoms in an RDKit molecule'''
    getter_type = RDPROP_GETTERS[prop_type]
    return [
        getattr(atom, getter_type)(prop)
            for atom in rdmol.GetAtoms()
    ]

# PROPERTY REMOVAL FUNCTIONS
@optional_in_place
def clear_atom_props(rdmol : RDMol) -> None:
    '''Wipe properties of all atoms in a molecule'''
    for atom in rdmol.GetAtoms():
        for prop_name in atom.GetPropNames():
            atom.ClearProp(prop_name)