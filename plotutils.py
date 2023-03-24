# Numeric processing and plotting
import numpy as np
import matplotlib.pyplot as plt


def scatter_3D(array : np.array) -> None:
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
