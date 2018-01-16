# cleardemand-heatmaps
This repository contains the code used to create an interactive bokeh heatmap for cleardemand data.

This code runs off Bokeh 0.12.13, which can be installed from Bokeh's documentation [here](https://bokeh.pydata.org/en/latest/docs/installation.html)

Once bokeh is installed, data must be stored in the base directory in a csv file. The code defaults to looking to a file called "PriceBook.csv" on the path. To generate the heatmap, run

`bokeh serve --show heatmap`

from the repository's root directory. This will open a web page with the interactive heatmap.

## heatmap
This folder contains the code to run the heatmap app through Bokeh. It can be run
by entering the command

`bokeh serve --show heatmap`

from the main directory, assuming it can access a csv file called 'PriceBook.csv'.

### heatmap.py
This file contains the two classes used by the heatmap
- HeatGrid -- class used to store the data which will populate the heatmap
- HeatMap -- class used to store display options for the heatmap

### main.py
This file is run to generate and populate the bokeh heatmap application. It builds the tools used for display, and instantiates the heatmap objects

## data_processing.py
This file contains functions used for data processing. Currently, this includes:
- small_load -- used to load a small number of the columns into a dataframe
- convert_dol_to_num -- used to convert columns in a dataframe which start with a $ to floats
- fix_dol -- used by convert_dol_to_num to identify and change necessary strings
- main -- accepts a file location, defaults to 'PriceBook.csv', and builds a dataframe with dollar columns converted to numerical data, called by the bokeh interactive heatmap

## clustering.py
This file contains functions used for clustering models and analysis. Not currently used by the interactive Bokeh heatmap. Currently, this includes:
- kmeans_cluster -- basic function to make a kmeans clusters
- hierarchical_cluster -- basic functionn to make a hierarchical clusters
- elbow_plot -- function to make an elbow plot of a KMeans clusters to determine proper number of clusters
- perform_pca -- basic function to do pca on a pandas dataframe

## plotter.py
This file contains functions used for saving basic plots. Not used by the interactive Bokeh heatmap. Currently, this includes:
- build_basic_heatmap -- function used to make a heatmap
- build_sorted_heatmap -- function used to make a crudely sorted heatmap
- correlation_plots -- function used to make correlation plots off different columns
- drop_nans -- function used to drop missing rows for correlation_plots
- check_dir -- function used to make new folders when necessary for plot storage locations
