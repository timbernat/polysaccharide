# Typing
from .. import molutils
from ..extratypes import RDMol
from . import imageutils, plotutils
from .named_colors import WHITE

# Plotting
import PIL
from matplotlib.pyplot import Figure, Axes
from matplotlib.colors import Normalize, Colormap
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
           

def set_rdkdraw_size(dim : int, aspect : float):
    '''Change image size and shape of RDMol images'''
    IPythonConsole.molSize = (int(aspect*dim), dim)   # Change image size

def rdmol_prop_heatmap(rdmol : RDMol, prop : str, cmap : Colormap, norm : Normalize, img_size : tuple[int, int]=(1_000, 1_000)) -> bytes: #IPyImage:
    '''Take a charged RDKit Mol and color atoms based on the magnitude of their charge'''
    colors = {
        atom.GetIdx() : cmap(norm(atom.GetDoubleProp(prop)))
            for atom in rdmol.GetAtoms()
    }
    atom_nums = [idx for idx in colors.keys()]

    draw = rdMolDraw2D.MolDraw2DCairo(*img_size) # or MolDraw2DCairo to get PNGs
    rdMolDraw2D.PrepareAndDrawMolecule(draw, rdmol, highlightAtoms=atom_nums, highlightAtomColors=colors)
    draw.FinishDrawing()
    img_bytes = draw.GetDrawingText()
    
    return imageutils.img_from_bytes(img_bytes)

def compare_chgd_rdmols(chgd_rdmol_1 : RDMol, chgd_rdmol_2 : RDMol, chg_method_1 : str, chg_method_2 : str, cmap : Colormap, flatten : bool=True) -> tuple[Figure, Axes]:
    '''Plot a labelled heatmap of the charge differences between 2 strucuturally identical RDKit Molecules with different partial charges'''
    if flatten:
        chgd_rdmol_1 = molutils.flattened_rmdol(chgd_rdmol_1)
        chgd_rdmol_2 = molutils.flattened_rmdol(chgd_rdmol_2)

    diff = molutils.difference_rdmol(chgd_rdmol_1, chgd_rdmol_2)
    vmin, vmax = diff.GetDoubleProp('DeltaPartialChargeMin'), diff.GetDoubleProp('DeltaPartialChargeMax'),
    norm = Normalize(vmin, vmax)
    ticks = (vmin, 0, vmax)

    image = rdmol_prop_heatmap(diff, prop='DeltaPartialCharge', cmap=cmap, norm=norm)
    image = imageutils.crop_borders(image, bg_color=WHITE)

    return plotutils.plot_image_with_colorbar(image, cmap, norm, label=f'\u0394q (elem. charge): {chg_method_1} vs {chg_method_2}', ticks=ticks)
