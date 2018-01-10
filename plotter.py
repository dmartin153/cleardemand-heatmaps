from __future__ import division
'''This module contains functions used for plotting'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def build_heatmap(df, index, column, value, sortby=None):
    '''This function builds a heatmap to look at the given value over various
    index and columns. Sorts by the sortby value, defaulting to the value.
    Inputs:
    df -- pandas dataframe with the data
    index -- column to use as the index of the pivot table
    column -- column to use as the columns of the pivot table
    value -- value to plot on the heatmap
    sortby -- value to use to sort the heatmap, defaults to value
    Outputs:
    None
    By: David Martin
    On: Jan 10, 2018'''
    if sortby==None:
        sortby=value
    name = '{val}_heatmap__{ind}_vs_{col}'.format(val=value, ind=index, col=column)
    pt = df.pivot_table(index=index, columns=column, values=[value,sortby], aggfunc=np.mean)
    main_column = np.argmax(df.groupby(column)[sortby].sum())
    main_index = np.argmax(df.groupby(index)[sortby].sum())
    pt.sort_values(by=main_column, axis=0, inplace=True)
    pt.sort_values(by=main_index, axis=1, inplace=True)
    fig = plt.figure()
    ax = sns.heatmap(pt)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_title(name.replace('_',' '))
    fig.tight_layout()
    saveloc='figures/heatmaps/'
    fig.savefig(saveloc+name+'.jpg')
    plt.close(fig)

def correlation_plots(df, x_col_name, y_col_name):
    '''This function makes scatter plots between the given columns, and provides
    correlation values'''
    n_df = drop_nans(df, [x_col_name, y_col_name])
    if not len(n_df[x_col_name]):
        print('no overlaping points between {} and {}'.format(x_col_name, y_col_name))
        return
    sns.set()
    x = n_df[x_col_name].values
    y = n_df[y_col_name].values
    saveloc = 'figures/correlation_plots/{}/'.format(x_col_name)
    check_dir(saveloc)
    name = '{}_vs_{}'.format(y_col_name, x_col_name)
    corr = np.corrcoef(x, y)[0][1]
    fig = plt.figure()
    plt.plot(x,y,'.',label='Correlation = {}'.format(corr), alpha=0.1)
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.title(name.replace('_', ' '))
    plt.legend()
    fig.savefig(saveloc+name+'.jpg')
    plt.close(fig)

def drop_nans(df, cols):
    '''This function returns a new dataframe with just the specified columns,
    dropping rows where there are no values'''
    n_df = df[cols].copy()
    return n_df.dropna()

def check_dir(location):
    '''This function checks if a location exists, and makes it if it doesn't.
    Only works for one layer.'''
    if not os.path.exists(location):
        os.makedirs(location)
