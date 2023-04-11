from collections import defaultdict
from pathlib import Path

from openff.toolkit import ForceField
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import LibraryChargeHandler

from .types import AtomIDMap, Accumulator, ResidueChargeMap
from .residues import ChargedResidue


# FUNCTIONS
def find_repr_residues(mol : Molecule) -> dict[str, int]:
    '''Determine names and smallest residue numbers of all unique residues in charged molecule
    Used as representatives for generating labelled SMARTS strings '''
    rep_res_nums = defaultdict(set) # numbers of representative groups for each unique residue, used to build SMARTS strings
    for atom in mol.atoms: 
        rep_res_nums[atom.metadata['residue_name']].add(atom.metadata['residue_number']) # collect unique residue numbers

    for res_name, ids in rep_res_nums.items():
        rep_res_nums[res_name] = min(ids) # choose group with smallest id of each residue to denote representative group

    return rep_res_nums

def get_averaged_charges(cmol : Molecule, monomer_data : dict[str, dict], distrib_mono_charges : bool=True, net_mono_charge : float=0.0) -> tuple[list[ChargedResidue], AtomIDMap]:
    '''Takes a charged molecule and a dict of monomer SMIRKS strings and averages charges for each repeating residue. 
    Returns a list of ChargedResidue objects, each of which holds:
        - A dict of the averaged charges by atom 
        - The name of the residue associated with the charges
        - A SMARTS string of the residue's structure
        - An nx.Graph representing the structure of the residue'''
    # rdmol = cmol.to_rdkit() # create rdkit representation of Molecule to allow for SMARTS generation
    mol_graph = cmol.to_networkx()
    rep_res_nums = find_repr_residues(cmol) # determine ids of representatives of each unique residue

    atom_id_mapping   = defaultdict(lambda : defaultdict(int))
    res_charge_accums = defaultdict(lambda : defaultdict(Accumulator))
    for atom in cmol.atoms: # accumulate counts and charge values across matching subsftructures
        res_name, res_num     = atom.metadata['residue_name'   ], atom.metadata['residue_number']
        substruct_id, atom_id = atom.metadata['substructure_id'], atom.metadata['pdb_atom_id'   ]

        if res_num == rep_res_nums[res_name]: # if atom is member of representative group for any residue...
            # rdmol.GetAtomWithIdx(atom_id).SetAtomMapNum(atom_id)  # ...and set atom number for labelling in SMARTS string
            atom_id_mapping[res_name][atom_id] = (substruct_id, atom.symbol) # ...collect pdb id...

        curr_accum = res_charge_accums[res_name][substruct_id] # accumulate charge info for averaging
        curr_accum.sum += atom.partial_charge.magnitude # eschew units (easier to handle, added back when writing to XML)
        curr_accum.count += 1

    avg_charges_by_residue = []
    for res_name, charge_map in res_charge_accums.items():
        # rdSMARTS = rdmolfiles.MolFragmentToSmarts(rdmol, atomsToUse=atom_id_mapping[res_name].keys()) # determine SMARTS for the current residue's representative group
        # mol_frag = rdmolfiles.MolFromSmarts(rdSMARTS) # create fragment from rdkit SMARTS to avoid wild atoms (using rdkit over nx.subgraph for more detailed atomwise info)
        
        SMARTS = monomer_data['monomers'][res_name] # extract SMARTS string from monomer data
        charge_map = {substruct_id : accum.average for substruct_id, accum in charge_map.items()} 
        atom_id_map = atom_id_mapping[res_name]

        mol_frag = mol_graph.subgraph(atom_id_map.keys()) # isolate subgraph of residue to obtain connectivity info for charge redistribution
        for atom_id, (substruct_id, symbol) in atom_id_map.items(): # assign additional useful info not present by default in graph
            mol_frag.nodes[atom_id]['substruct_id'] = substruct_id
            mol_frag.nodes[atom_id]['symbol'] = symbol

        chgd_res = ChargedResidue(
            charges=charge_map,
            residue_name=res_name,
            SMARTS=SMARTS,
            mol_fragment=mol_frag
        )
        if distrib_mono_charges: # only distribute charges if explicitly called for (enabled by default)
            chgd_res.distrib_mono_charges(desired_net_charge=net_mono_charge)
        avg_charges_by_residue.append(chgd_res)

    return avg_charges_by_residue, atom_id_mapping

def write_lib_chgs_from_mono_data(monomer_data : dict[str, dict], offxml_src : Path, output_path : Path) -> tuple[ForceField, list[LibraryChargeHandler]]: # TODO - refactor to support using ResidueChargeMap for charges
    '''Takes a monomer JSON file (must contain charges!) and a force field XML file and appends Library Charges based on the specified monomers. Outputs to specified output_path'''
    assert(output_path.suffix == '.offxml') # ensure output path is pointing to correct file type
    assert(monomer_data.get('charges') is not None) # ensure charge entries are present

    forcefield = ForceField(offxml_src) # simpler to add library charges through forcefield API than to directly write to xml

    lc_handler = forcefield["LibraryCharges"]
    lib_chgs = [] #  all library charges generated from the averaged charges for each residue
    for resname, charge_dict in monomer_data['charges'].items(): # ensures no uncharged structure are written as library charges (may be a subset of the monomers structures in the file)
        lc_entry = { # stringify charges into form usable for library charges
            f'charge{cid}' : f'{charge} * elementary_charge' 
                for cid, charge in charge_dict.items()
        } 

        lc_entry['smirks'] = monomer_data['monomers'][resname] # add SMIRKS string to library charge entry to allow for correct labelling
        lc_params = LibraryChargeHandler.LibraryChargeType(allow_cosmetic_attributes=True, **lc_entry) # must enable cosmetic params for general kwarg passing
        
        lc_handler.add_parameter(parameter=lc_params)
        lib_chgs.append(lc_params)  # record library charges for reference
    forcefield.to_file(output_path) # write modified library charges to new xml (avoid overwrites in case of mistakes)
    
    return forcefield, lib_chgs