'''This module contains functions used for plotting'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def build_heatmap(df, index, column, value, sortby=None):
    '''This function builds a heatmap to look at the
    given value over various index and columns'''
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
