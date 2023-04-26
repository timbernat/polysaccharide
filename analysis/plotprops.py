from ..graphics.plotutils import presize_subplots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rdfs(dframe : pd.DataFrame, **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    '''Takes a DataFrame populated with RDF data and generates sequential plots'''
    radii = dframe.filter(regex='Radius')
    radii_label = radii.columns[0]
    rdfs  = dframe.filter(regex='g\(r\)') # need escapes to treat parens as literals

    fig, ax = presize_subplots(nrows=1, ncols=len(rdfs.columns), **plot_kwargs)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax]) # convert singleton subplots into arrays so that they don;t break when attempting to be flattened

    for curr_ax, (name, rdf) in zip(ax.flatten(), rdfs.items()):
        curr_ax.plot(radii, rdf)
        curr_ax.set_xlabel(radii_label)
        curr_ax.set_ylabel(name)

    return fig, ax

def plot_time_props(dframe : pd.DataFrame, **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    '''Takes a DataFrame populated with polymer time series property data and generates sequential plots'''
    times = dframe.filter(regex='Time')
    times_label = times.columns[0]
    prop_data = dframe[[label for label in dframe.columns if label != times_label]] # index props with all non-time point columns

    fig, ax = presize_subplots(nrows=1, ncols=len(prop_data.columns), **plot_kwargs)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax]) # convert singleton subplots into arrays so that they don;t break when attempting to be flattened

    for curr_ax, (name, rdf) in zip(ax.flatten(), prop_data.items()):
        curr_ax.plot(times, rdf)
        curr_ax.set_xlabel(times_label)
        curr_ax.set_ylabel(name)

    return fig, ax