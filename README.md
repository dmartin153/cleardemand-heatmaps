# cleardemand-heatmaps
This repository contains the code used to create an interactive bokeh heatmap for cleardemand data.

This code runs off Bokeh 0.12.13, which can be installed from Bokeh's documentation [here](https://bokeh.pydata.org/en/latest/docs/installation.html)

Once bokeh is installed, data must be stored in the base directory in a csv file. The code defaults to looking to a file called "PriceBook.csv" on the path. To generate the heatmap, simple run

`bokeh serve --show heatmap`

from the repository's root directory. This will open a web page with the interactive heatmap.

## data_processing.py
This file contains functions used for data processing. Currently, this includes:
- small_load -- used to load a small number of the columns into a dataframe
- convert_dol_to_num -- used to convert columns in a dataframe which start with a $ to floats
- fix_dol -- used by convert_dol_to_num to identify and change necessary strings

## clustering.py
This file contains functions used for clustering models and analysis. Not currently used by the interactive Bokeh heatmap. Currently, this includes:
- kmeans_cluster -- basic function to make a kmeans clusters
- hierarchical_cluster -- basic functionn to make a hierarchical clusters
- elbow_plot -- function to make an elbow plot of a KMeans clusters to determine proper number of clusters
- perform_pca -- basic function to do pca on a pandas dataframe

## plotter.py
This file contains functions used for saving basic plots. Not used by the interactive Bokeh heatmap. Currently, this includes:
- build_heatmap -- function used to make a heatmap

## heatmap
This folder contains the code to run the heatmap app through Bokeh. It can be run
by entering the command

`bokeh serve --show heatmap`

from the main directory, assuming it can access an appropriate csv file.
