'''This module contains code related to clustering models'''
import numpy as np
import pandas as pd
import pdb
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def kmeans_cluster(df,cols=['RecPrice'],n_clusters=3):
    '''This function builds and returns a basic Kmeans cluster for the provided
    dataframe and columns.
    Inputs:
    df -- pandas dataframe with the data
    cols -- array of columns to use for clustering, defaults to using just the Cost column
    Outputs:
    model -- a fit kmeans cluster built with the provided parameters'''
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    scaler = StandardScaler()
    X = np.array(df[cols])
    X = scaler.fit_transform(X)
    kmeans.fit(X)
    return kmeans

def elbow_plot(df,cols=['Cost']):
    '''This function makes an elbow plot to evaluate the optimal number of clusters
    using Within Cluster Sum of Squares
    Inputs:
    df -- pandas dataframe with the data
    cols -- array of columns to use for clustering, defaults to using just the Cost column
    Outputs:
    None'''
    sns.set
    wcss = []
    scaler = StandardScaler()
    X = np.array(df[cols])
    X = scaler.fit_transform(X)
    for i in range(1,20):
        model = KMeans(n_clusters=i,random_state=1)
        model.fit(X)
        wcss.append(model.inertia_)
    fig = plt.figure()
    plt.plot(range(1,20), wcss,'.')
    plt.xlabel('Clusters (#)')
    plt.ylabel('WCSS')
    fig.show()

def hierarchical_cluster(df,cols=['Cost'], n_clusters=3):
    '''This function builds and returns a basic hierarchical cluster for the provided
    dataframe and columns.
    Inputs:
    df -- pandas dataframe with the  data
    cols -- array of  columns to use for clustering, defaults to using just the Cost column
    Outputs:
    model -- a fit Agglomerative cluster built with the provided parameters'''
    hier = AgglomerativeClustering(n_clusters=n_clusters)
    scaler = StandardScaler()
    X = np.array(df[cols])
    X = scaler.fit_transform(X)
    hier.fit(X)
    return hier

def perform_pca(df,cols=None,n_components=3):
    '''this function performs a pca on the provided dataframe.
    Inputs:
    df -- dataframe to have pca performed on
    cols -- columns to extract data from. If none (default), then use all ints and floats
    n_components -- number of principle components to analyze
    Outputs:
    pca -- a fit pca model'''
    pca = PCA(n_components=n_components)
    if cols:
        X = df[cols].values
    else:
        X = df.select_dtypes([int,float]).fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca.fit(X)
    return pca

def explained_variance(df,cols=None):
    '''this function makes a plot for explained total variance of the dataframe
    for varying numbers of pca components kept.'''
    pca = PCA(n_components=len(cols))
    scaler = StandardScaler()
    X = df[cols].values
    stand_X = scaler.fit_transform(X)
    pca.fit(stand_X)
    fig = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_),'.')
    plt.xlabel('Principle Components (#)')
    plt.ylabel('Expained Variance (proportion)')
    fig.show()
    return pca

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
