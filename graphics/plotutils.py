# Numeric processing and plotting
from typing import Callable, Optional
from pathlib import Path
from PIL.Image import Image

import numpy as np
from pandas import DataFrame
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.colors import Normalize, Colormap
from matplotlib.colorbar import Colorbar, ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

ColorMapper = Callable[[float], tuple[int, ...]]


# GENERAL PLOTTING FUNCTIONS
def scatter_3D(array : np.array) -> None:
    '''Plot an Nx3 array of (x, y, z) coordinate sets in 3-D'''
    x, y, z = array.T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)

def presize_subplots(nrows : int, ncols : int, scale : float=15.0, elongation : float=1.0) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    '''
    Prepare a grid of predetermined number of matplotlib subplot axes of a particular size and aspect ratio
    Returns the resulting Figure and array of individual subplot Axes 
    '''
    aspect = (nrows / ncols) * elongation
    return plt.subplots(nrows, ncols, figsize=(scale, aspect*scale))

def plot_df_props(x_dframe : DataFrame, y_dframe : DataFrame, nrows : int=None, ncols : int=None, **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    '''Takes a DataFrame populated with polymer time series property data and generates sequential plots'''
    num_series = len(y_dframe.columns)
    if (nrows is None) and (ncols is None): # TODO : deprecate this terribleness in favor of smart axis sizing to aspect ratio
        nrows = 1
        ncols = num_series
    if (nrows is None):
        nrows = ceil(num_series / ncols)
    if (ncols is None):
        ncols = ceil(num_series / nrows)
    assert(nrows * ncols) >= num_series

    fig, ax = presize_subplots(nrows=nrows, ncols=ncols, **plot_kwargs)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax]) # convert singleton subplots into arrays so that they don;t break when attempting to be flattened

    for curr_ax, (name, y_dframe) in zip(ax.flatten(), y_dframe.items()):
        curr_ax.plot(x_dframe, y_dframe)
        curr_ax.set_xlabel(x_dframe.columns[0])
        curr_ax.set_ylabel(name)

    return fig, ax

# COLORBAR FUNCTIONS
def make_cmapper(cmap_name : str, vmin : float, vmax : float) -> ColorMapper:
    '''Wrapper for making normalized color map function'''
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin, vmax)

    def cmapper(val_to_map : float) -> tuple[int, ...]:
        '''Actual colormapping function'''
        return cmap(norm(val_to_map))
    
    return cmapper

def plot_image_with_colorbar(image : Image, cmap : Colormap, norm : Normalize, label : str='', ticks : Optional[list[float]]=None, dim : int=8) -> tuple[plt.Figure, plt.Axes]:
    '''Plots a PIL image with a colorbar and norm of ones choice'''
    fig, ax = plt.subplots(figsize=(dim, dim))

    mpim = ax.imshow(image, cmap=cmap, norm=norm)
    width, height = image.size
    if height > width:
        cax_loc, orient = ('right', 'vertical')
    else:
        cax_loc, orient = ('bottom', 'horizontal')

    div = make_axes_locatable(ax) # allow for creatioin of separate colorbar axis
    ax.set_axis_off() # prevent ticks from interrupting image
    cax = div.append_axes(cax_loc, size='5%', pad='2%')
    cbar = fig.colorbar(mpim, cax=cax, label=label, ticks=ticks, orientation=orient)

    return fig, ax

def draw_colorbar(cmap_name : str, vmin : float, vmax : float, label : str, save_path : Path=None) -> Colorbar:
    '''Create a matplotlib colorbar object with appropriate norm, color scale, and labels'''
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin, vmax)
    
    fig = plt.figure()
    ax = fig.add_axes([0.9, 0.1, 0.05, 0.95]) # TODO : generalize this sizing
    ticks = [vmin, 0, vmax]

    cbar = ColorbarBase(ax, orientation='vertical', cmap=cmap, norm=norm, ticks=ticks, label=label)
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close()

    return cbar
