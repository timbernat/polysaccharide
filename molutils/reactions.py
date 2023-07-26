'''Tools for defining, modelling, and executing chemical reactions'''
# Custom Imports
from .rdmol import rdprops, rdbond, rdlabels
from .rdmol.rdtypes import RDMol

# Generic Imports
from functools import cached_property
from itertools import combinations, chain

# Typing and Subclassing
from typing import ClassVar, Iterable, Optional, Union
from dataclasses import dataclass, field

# Cheminformatics
from rdkit import Chem
from rdkit.Chem import rdChemReactions


# REACTION INFORMATICS CLASSES
@dataclass
class RxnProductInfo:
    '''For storing atom map numbers associated with product atoms and bonds participating in a reaction'''
    prod_num : int
    reactive_atom_map_nums : list[int] = field(default_factory=list)

    new_bond_ids_to_map_nums : dict[int, tuple[int, int]] = field(default_factory=dict)
    mod_bond_ids_to_map_nums : dict[int, tuple[int, int]] = field(default_factory=dict)
    
class AnnotatedReaction(rdChemReactions.ChemicalReaction):
    '''
    RDKit ChemicalReaction with additional useful information about product atom and bond mappings

    Initialization must be done either via AnnotatedReaction.from_smarts or AnnotatedReaction.from_rdmols,
    asdirect override of pickling in __init__ method not yet implemented
    '''
    @classmethod
    def from_smarts(cls, rxn_smarts : str) -> 'AnnotatedReaction':
        '''For instantiating reactions directly from molecules instead of SMARTS strings'''
        rdrxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)

        return cls(rdrxn) # pass to init method

    @classmethod
    def from_rdmols(cls, reactant_templates : Iterable[RDMol], product_templates : Iterable[RDMol]) -> 'AnnotatedReaction':
        '''For instantiating reactions directly from molecules instead of SMARTS strings'''
        react_str = '.'.join(Chem.MolToSmarts(react_template) for react_template in reactant_templates)
        prod_str  = '.'.join(Chem.MolToSmarts(prod_template) for prod_template in product_templates)
        rxn_smarts = f'{react_str}>>{prod_str}'

        return cls.from_smarts(rxn_smarts)

    @cached_property
    def reacting_atom_map_nums(self) -> list[int]:
        '''List of the map numbers of all reactant atoms which participate in the reaction'''
        return [
            self.GetReactantTemplate(reactant_id).GetAtomWithIdx(atom_id).GetAtomMapNum()
                for reactant_id, reacting_atom_ids in enumerate(self.GetReactingAtoms())
                    for atom_id in reacting_atom_ids
        ]
    
    @cached_property
    def map_nums_to_reactant_nums(self) -> dict[int, int]:
        '''Back-map yielding the index of the source reactant for the atom of each map number'''
        return {
            atom.GetAtomMapNum() : i
                for i, react_template in enumerate(self.GetReactants())
                    for atom in react_template.GetAtoms()
        }
    
    @cached_property
    def product_info_maps(self) -> dict[int, RxnProductInfo]:
        '''Map from product index to information about reactive atoms and bonds in that product'''
        # map reacting atoms and bonds for each product
        prod_info_map = {}
        for i, product_template in enumerate(self.GetProducts()):
            prod_info = RxnProductInfo(i)
            prod_info.reactive_atom_map_nums = [
                map_num
                    for map_num in self.reacting_atom_map_nums
                        if map_num in rdlabels.get_ordered_map_nums(product_template)
            ]

            for bond_id, atom_id_pair in rdbond.get_bonded_pairs_by_map_nums(product_template, *prod_info.reactive_atom_map_nums).items(): # consider each pair of reactive atoms
                map_num_1, map_num_2 = map_num_pair = tuple(
                    product_template.GetAtomWithIdx(atom_id).GetAtomMapNum()
                        for atom_id in atom_id_pair
                )
                
                if self.map_nums_to_reactant_nums[map_num_1] == self.map_nums_to_reactant_nums[map_num_2]: # if reactant IDs across bond match, the bond must have been modified (i.e. both from single reactant...)
                    prod_info.mod_bond_ids_to_map_nums[bond_id] = map_num_pair
                else: # otherwise, bond must be newly formed (spans between previously disjoint monomers) 
                    prod_info.new_bond_ids_to_map_nums[bond_id] = map_num_pair
            prod_info_map[i] = prod_info
        
        return prod_info_map
    

# REACTOR (EXECUTION) CLASSES
@dataclass
class Reactor:
    '''Class for executing a reaction template on collections of RDMol "reactants"'''
    rxn_schema : AnnotatedReaction
    reactants : Iterable[RDMol]
    products  : Optional[Iterable[RDMol]] = field(init=False, default=None)
    _has_reacted : bool = field(init=False, default=False)

    _ridx_prop_name : ClassVar[str] = field(init=False, default='reactant_idx') # name of the property to assign reactant indices to; set for entire class

    # PRE-REACTION PREPARATION METHODS
    def _activate_reaction(self) -> None:
        '''Check that the reaction schema provided is well defined and initialized'''
        pass

    def _label_reactants(self) -> None:
        '''Assigns "reactant_idx" Prop to all reactants to help track where atoms go during the reaction'''
        for i, reactant in enumerate(self.reactants):
            for atom in reactant.GetAtoms():
                atom.SetIntProp(self._ridx_prop_name, i)

    def __post_init__(self) -> None:
        '''Pre-processing of reaction and reactant Mols'''
        self._activate_reaction()
        self._label_reactants()

    # POST-REACTION CLEANUP METHODS
    @classmethod
    def _relabel_reacted_atoms(cls, product : RDMol, reactant_map_nums : dict[int, int]) -> None:
        '''Re-assigns "reactant_idx" Prop to modified reacted atoms to re-complete atom-to-reactant numbering'''
        for atom_id in rdprops.atom_ids_with_prop(product, 'old_mapno'):
            atom = product.GetAtomWithIdx(atom_id)
            map_num = atom.GetIntProp('old_mapno')

            atom.SetIntProp(cls._ridx_prop_name, reactant_map_nums[map_num])
            atom.SetAtomMapNum(map_num) # TOSELF : in future, might remove this (makes mapping significantly easier, but is ugly for labelling)

    @staticmethod
    def _sanitize_bond_orders(product : RDMol, product_template : RDMol, product_info : RxnProductInfo) -> None:
        '''Ensure bond order changes specified by the reaction are honored by RDKit'''
        for prod_bond_id, map_num_pair in product_info.mod_bond_ids_to_map_nums.items():
            target_bond = product_template.GetBondWithIdx(prod_bond_id)

            product_bond = rdbond.get_bond_by_map_num_pair(product, map_num_pair)
            # product_bond = product.GetBondBetweenAtoms(*rdlabels.atom_ids_by_map_nums(product, *map_num_pair))
            assert(product_bond.GetBeginAtom().HasProp('_ReactionDegreeChanged')) 
            assert(product_bond.GetEndAtom().HasProp('_ReactionDegreeChanged')) # double check that the reaction agrees that the bond has changed

            product_bond.SetBondType(target_bond.GetBondType()) # set bond type to what it *should* be from the reaction schema

    # REACTION EXECUTION METHODS
    def react(self, repetitions : int=1, clear_props : bool=False) -> None:
        '''Execute reaction and generate product molecule'''
        self.products = [
            product # unroll nested tuples that RDKit provides as reaction products
                for product in chain.from_iterable(self.rxn_schema.RunReactants(self.reactants, maxProducts=repetitions))
        ]
        self._has_reacted = True # set reaction flag

        # post-reaction cleanup
        for i, product in enumerate(self.products): # TODO : generalize to work when more than 1 repetition is requested
            self._relabel_reacted_atoms(product, self.rxn_schema.map_nums_to_reactant_nums)
            self._sanitize_bond_orders(product,
                product_template=self.rxn_schema.GetProductTemplate(i),
                product_info=self.rxn_schema.product_info_maps[i]
            )
            if clear_props:
                rdprops.clear_atom_props(product)

@dataclass
class CondensationReactor(Reactor):
    '''Special case of Reactor with two reactant species forming one product'''
    def __post_init__(self) -> None:
        assert(self.rxn_schema.GetNumReactantTemplates() == 2)
        assert(self.rxn_schema.GetNumProductTemplates() == 1)

        return super().__post_init__()
    
    @property
    def product(self) -> RDMol:
        return self.products[0]
    
    @property
    def product_info(self) -> RDMol:
        return self.rxn_schema.product_info_maps[0]

@dataclass
class PolymerizationReactor(CondensationReactor):
    '''Reactor which handles monomer partitioning post-polymerization condensation reaction'''
    def inter_monomer_bond_candidates(self, valid_backbone_atoms : tuple[str]=('C', 'N', 'O')) -> list[int]:
        '''Returns the bond index of the most likely candidate for a newly-formed bond in the product which was formed between the reactants
        Can optionally define which atoms are valid as main-chain atoms (by default just CNO)'''
        possible_bridgehead_ids = [ # determine all atomic positions which are: 
            atom_id
                for atom_id in rdprops.atom_ids_with_prop(self.product, 'was_dummy')           # 1) former ports (i.e. outside of monomers)
                    if self.product.GetAtomWithIdx(atom_id).GetSymbol() in valid_backbone_atoms # 2) valid backbone atoms (namely, not hydrogens)
        ]
        
        return [
            new_bond_id
                for new_bond_id in self.product_info.new_bond_ids_to_map_nums.keys()  # for each newly formed bond...
                    for bridgehead_id_pair in combinations(possible_bridgehead_ids, 2) # ...find the most direct path between bridgehead atoms...
                        if new_bond_id in rdbond.get_shortest_path_bonds(self.product, *bridgehead_id_pair) # ...and check if the new bond lies along it
        ]

    def polymerized_fragments(self, separate : bool=True) -> Union[RDMol, tuple[RDMol]]:
        '''Cut product on inter-monomer bond, returning the resulting fragments'''
        clean_product = rdlabels.clear_atom_map_nums(self.product, in_place=False)
        fragments = Chem.FragmentOnBonds(clean_product, bondIndices=[self.inter_monomer_bond_candidates()[0]])

        if separate:
            return Chem.GetMolFrags(fragments, asMols=True)
        return fragments # if separation is not requested, return as single fragmented molecule object