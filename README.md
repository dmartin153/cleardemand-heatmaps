# cleardemand-heatmaps
This repository contains the code used for clustering and making heatmaps of cleardemand data

##data_processing.py
This file contains functions used for data processing. Currently, this includes:
- small_load -- used to load a small number of the columns into a dataframe
- convert_dol_to_num -- used to convert columns in a dataframe which start with a $ to floats
- fix_dol -- used by convert_dol_to_num to identify and change necessary strings

##clustering.py
This file contains functions used for clustering models and analysis. currently, this includes:
- kmeans_cluster -- basic function to make a kmeans clusters
- hierarchical_cluster -- basic functionn to make a hierarchical clusters
- elbow_plot -- function to make an elbow plot of a KMeans clusters to determine proper number of clusters
- perform_pca -- basic function to do pca on a pandas dataframe
