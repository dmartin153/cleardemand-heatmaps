from __future__ import division
'''This module contains functions used for plotting'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pdb
import evaluation
import itertools
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import data_processing


def build_basic_heatmap(df, index, column, value):
    '''This function builds a heatmap to look at the given value over various
    index and columns. Sorts by the value.
    Inputs:
    df -- pandas dataframe with the data
    index -- column to use as the index of the pivot table
    column -- column to use as the columns of the pivot table
    value -- value to plot on the heatmap, also what the heatmap is sorted by
    Outputs:
    None
    Saves a jpeg in figures/basic_heatmaps/
    By: David Martin
    On: Jan 10, 2018'''
    sns.set()
    name = '{val}_heatmap__{ind}_vs_{col}'.format(val=value, ind=index, col=column)
    pt = df.pivot_table(index=index, columns=column, values=value, aggfunc=np.mean)
    main_column = np.argmax(df.groupby(column)[value].var())
    main_index = np.argmax(df.groupby(index)[value].var())
    pt.sort_values(by=main_column, axis=0, inplace=True, ascending=True, na_position='first')
    pt.sort_values(by=main_index, axis=1, inplace=True, ascending=True, na_position='first')
    fig = plt.figure()
    ax = sns.heatmap(pt)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_title(name.replace('_',' '))
    fig.tight_layout()
    saveloc='figures/basic_heatmaps/'
    check_dir(saveloc)
    fig.savefig(saveloc+name+'.jpg')
    plt.close(fig)

def build_sorted_heatmap(df, index, column, value, sortby=None):
    '''This function builds a heatmap to look at the given value over various
    index and columns. Sorts by the sortby column. Defaults to the value.
    Inputs:
    df -- pandas dataframe with the data
    index -- column to use as the index of the pivot table
    column -- column to use as the columns of the pivot table
    value -- value to plot on the heatmap, also what the heatmap is sorted by
    sortby -- column to use to sort the heatmap
    Outputs:
    None
    Saves a jpeg in figures/sorted_heatmaps/
    By: David Martin
    On: Jan 10, 2018'''
    sns.set()
    name = '{val}_heatmap__{ind}_vs_{col}_by_{sort}'.format(val=value, ind=index, col=column, sort=sortby)
    pt = df.pivot_table(index=index, columns=column, values=[value,sortby], aggfunc=np.mean)
    main_column = np.argmax(df.groupby(column)[value].var())
    main_index = np.argmax(df.groupby(index)[value].var())
    pt.sort_values(by=(sortby,main_column), axis=0, inplace=True, ascending=True, na_position='first')
    pt.sort_values(by=main_index, axis=1, inplace=True, ascending=True, na_position='first')
    fig = plt.figure()
    ax = sns.heatmap(pt[value])
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_title(name.replace('_',' '))
    fig.tight_layout()
    saveloc='figures/sorted_heatmaps/'
    check_dir(saveloc)
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

def find_labeled_points(label,df):
    '''This function makes a list of potential revenues or profits using the
    clustering done in 'cluster_labels' and values in the label passed in'''
    row_index = df.index
    clusters = df['cluster_labels'].unique()
    keypoints = df[label]
    outputs = []
    for permut_index in itertools.product(range(0,len(keypoints[0])), repeat=len(clusters)):
        val = 0
        for per_ind in range(0,len(permut_index)):
            clusterinds = df[df['cluster_labels'] == clusters[per_ind]].index
            val += df.loc[clusterinds, label].apply(lambda x: x[permut_index[per_ind]]).sum()
        outputs.append(val)
    return outputs

def plot_ppf(df,title=None):
    '''this function plots a basic production possibility frontier'''
    sns.set()
    profits = find_labeled_points('KeyProfitPoints',df)
    revs = find_labeled_points('KeyRevPoints',df)
    fig = plt.figure()
    plt.plot(revs,profits, 'g.', alpha=0.5, label='Strategy Variants')
    plt.plot(df['CurRev'].sum(), df['CurProfit'].sum(), 'r.', label='Current Prices')
    plt.plot(df['RecRev'].sum(), df['RecProfit'].sum(), 'b.', label='Recommended Prices')
    plt.xlabel('Total Revenue')
    plt.ylabel('Total Profit')
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    return fig

def cluster_fake_ppf(df, columns=['CurRev'], n_clusters = 2, title=None):
    kmeans = KMeans(n_clusters = n_clusters, random_state = 20)
    X = np.array(df[columns])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kmeans.fit(X)
    df['cluster_labels'] = kmeans.labels_
    fig = plot_ppf(df, title)
    return fig

def make_and_save_ppf(df,name,cluster_cols=['CurRev'],n_clusters=3,n_strats=8):
    saveloc = 'figures/PPFs/'
    check_dir(saveloc)
    plot_name = name+'_{}strats_{}clusters_of_{}'.format(n_strats,n_clusters,'_'.join(cluster_cols))
    n_df = df.copy()
    data_processing.add_key_points(n_df,strategies=n_strats)
    fig = cluster_fake_ppf(n_df, columns=cluster_cols, n_clusters=n_clusters, title=plot_name)
    fig.savefig(saveloc+plot_name+'.jpg')
    plt.close(fig)
