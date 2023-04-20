# Typing
from ..molutils.rdmol import rdcompare, rdconvert
from ..general import GREEK_UPPER
from ..extratypes import RDMol
from . import imageutils, plotutils
from .named_colors import WHITE

# Plotting
import PIL
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, Colormap
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps, MolsToGridImage, IPythonConsole
           

def set_rdkdraw_size(dim : int, aspect : float):
    '''Change image size and shape of RDMol images'''
    IPythonConsole.molSize = (int(aspect*dim), dim)   # Change image size

def rdmol_prop_heatmap(rdmol : RDMol, prop : str, cmap : Colormap, norm : Normalize, annotate : bool=False, precision : int=5, img_size : tuple[int, int]=(1_000, 1_000)) -> bytes: #IPyImage:
    '''Take a charged RDKit Mol and color atoms based on the magnitude of their charge'''
    colors, prop_vals, atom_nums = {}, [], []
    for atom in rdmol.GetAtoms():
        atom_num, prop_val = atom.GetIdx(), atom.GetDoubleProp(prop)
        colors[atom_num] = cmap(norm(prop_val))
        atom_nums.append(atom_num)
        prop_vals.append(prop_val)

        if annotate:
            atom.SetProp('atomNote', str(round(prop_val, precision))) # need to convert to string, as double is susceptible to float round display errors (shows all decimal places regardless of rounding)

    draw = rdMolDraw2D.MolDraw2DCairo(*img_size) # or MolDraw2DCairo to get PNGs
    rdMolDraw2D.PrepareAndDrawMolecule(draw, rdmol, highlightAtoms=atom_nums, highlightAtomColors=colors)
    draw.FinishDrawing()
    img_bytes = draw.GetDrawingText()
    
    return imageutils.img_from_bytes(img_bytes)

def compare_chgd_rdmols(chgd_rdmol_1 : RDMol, chgd_rdmol_2 : RDMol, chg_method_1 : str, chg_method_2 : str, cmap : Colormap=plt.get_cmap('turbo'),
                         flatten : bool=True, converter : Union[str, rdconvert.RDConverter]='SMARTS', **heatmap_args) -> tuple[plt.Figure, plt.Axes]:
    '''Plot a labelled heatmap of the charge differences between 2 strucuturally identical RDKit Molecules with different partial charges'''
    if flatten:
        chgd_rdmol_1 = rdcompare.flattened_rdmol(chgd_rdmol_1, converter=converter)
        chgd_rdmol_2 = rdcompare.flattened_rdmol(chgd_rdmol_2, converter=converter)

    diff = rdcompare.difference_rdmol(chgd_rdmol_1, chgd_rdmol_2)
    vmin, vmax = diff.GetDoubleProp('DeltaPartialChargeMin'), diff.GetDoubleProp('DeltaPartialChargeMax'),
    norm = Normalize(vmin, vmax)
    ticks = (vmin, 0, vmax)

    image = rdmol_prop_heatmap(diff, prop='DeltaPartialCharge', cmap=cmap, norm=norm, **heatmap_args)
    image = imageutils.crop_borders(image, bg_color=WHITE)

    return plotutils.plot_image_with_colorbar(image, cmap, norm, label=f'{GREEK_UPPER["delta"]}q (elem. charge): {chg_method_1} vs {chg_method_2}', ticks=ticks)
