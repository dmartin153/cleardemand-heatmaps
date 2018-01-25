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
- main -- accepts a file location, defaults to 'PriceBook.csv', builds a dataframe with dollar columns converted to numerical data, adds q values if they are missing, and runs drop_rows and add_price_variation
- drop_rows -- removes products from a data frame which are not in all the areas
- build_num_prices -- adds columns of "NumIdenticalPrices" and "NumTotalProducts" which identify the number of identical prices a product has and the total number of each product in the dataframe
- build_success_metrics -- creates columns of "NormAreaProfit", "NormAreaRev", "Strategy", "RegretRev", "RegretProfit", "NormProductProfit", and "NormProductRev"
- add_q -- calculates and adds a columns of "Q" to the dataframe
- add_key_points -- this function builds columns of "KeyProfitPoints" and "KeyRevPoints" for the number of strategies specified (used for computational solution to efficient frontier)
- add_price_variation -- this function adds columns of "CurPriceVariation" and "CurPriceStdVariation" to the provided dataframe
- build_isoforest_preds -- adds columns of "IsoForestPredict_std", "IsoForestPredict_var", "IsoForestPredict_stdvar", "IsoForestPredict_std_toprev", "IsoForestPredict_var_toprev", "IsoForestPredict_stdvar_toprev", "IsoForestPrice", "IsoForestPredict_ensemble", and "FullAutoPricing".


## clustering.py
This file contains functions used for clustering models and analysis. Not currently used by the interactive Bokeh heatmap. Currently, this includes:
- kmeans_cluster -- basic function to make a kmeans clusters
- hierarchical_cluster -- basic functionn to make a hierarchical clusters
- elbow_plot -- function to make an elbow plot of a KMeans clusters to determine proper number of clusters
- perform_pca -- basic function to do pca on a pandas dataframe
- explained_variance -- function to make a plot of total explained variance for varying numbers of pca kept
- find_labeled_points -- returns a list of values identified by the label parameter within the provided dataframe, spliting it by cluster_labels (requires clustering to work)

## plotter.py
This file contains functions used for saving basic plots. Not used by the interactive Bokeh heatmap. Currently, this includes:
- build_basic_heatmap -- function used to make a heatmap
- build_sorted_heatmap -- function used to make a crudely sorted heatmap
- correlation_plots -- function used to make correlation plots off different columns
- drop_nans -- function used to drop missing rows for correlation_plots
- check_dir -- function used to make new folders when necessary for plot storage locations
- plot_ppf -- plots a computationally computationally determined efficient frontier. returns the figure object
- cluster_fake_ppf -- Clusters along the indicated column into the n_clusters provided, and returns a figure of the computationally computed efficient frontier
- make_and_save_ppf -- performs all the computation required to calculate the computational efficient frontier and makes and saves a figure with the provided inputs
- dollar_v_price -- returns a figure of revenue and profit vs price for the product in the index provided for the dataframe passed in
- single_product_rev_v_profit -- returns a figure of profit vs revenue for the product in the dataframe indicated by the index. If bounds=1, also plots error bars
- efficient_frontier_plot -- returns a figure of Revenue vs Profit with the analytically derived efficient frontier for the provided dataframe
- plot_q -- plots how Q changes for a variety of prices for the product on the index provided for the dataframe provided

## evaluation.py
This file contains functions used to evaluate the revenue and profit of pricing options, including:
- calculate_revenue -- function to calculate the revenue of a product given the price, beta, and q value
- calculate_profit -- function to calculate the profit of a product given the price, beta, q, and cost
- rev_prof_gain -- function which returns the change in revenue and profit from an old price to a new price, given the beta, q, and cost
- calculate_available_rev_prof -- function which calculates the difference between the recommended and current revenue and profit given a dataframe

## modeling.py
This file contains functions used in key modeling steps
- feature_cleaning -- returns an X feature matrix and y target vector which drops nans using the dataframe provided, with features indicated by cols, and a target from the target
- build_regression_model -- returns a model, X_train, y_train, X_test, and y_test when provided with a dataframe, model to use, feature columns, target column, and random state
- iso_forest_predict_outliers -- returns a list of outlier indexes, using an isolation forest on the - build_new_suggested_prices -- creates a new column "IsoSugPrice" which is closer to the average price, using the outliers identified via the outliers input
- add_iso_sug_price -- runs iso_forest_predict_outliers and build_new_suggested_prices for the provided dataframe
- isoforestpred -- fits and returns the predictions of an isolation forest on the dataframe provided with the fit_option and training_option indicated

## presentation.py
This file contains scripts used to build the figures used in presentations
- main -- Runs all the functions for the provided save location and base data file location
- make_revenue_heatmap - returns a dataframe from the filelocation identified, after doing basic data processing and adding an isolation forest prediction
- make_revenue_heatmap -- makes and saves a heatmap of revenue with the normal sorting functions (Area on y axis, product on x axis, rows sorted by revenue, columns sorted by price)
- make_price_heatmap -- makes and saves a heatmap of current price with the nnormal sorting functions
- make_price_varstd_heatmap -- makes and saves a heatmap of current price standard deviation of variation with the normal sorting functions
- make_isoforest_heatmap -- makes and saves a heatmap of the ensemble of isolation forest predictions
- make_q_and_errors -- makes and saves a plot of Q vs price for the highest revenue index in the dataframe
- make_product_prof_rev -- makes and saves a plot of revenue and profit vs price for the highest revenue product in the dataframe
- make_product_frontier -- make and save a plot of revenue vs profit for varying prices for the highest revenue product in the dataframe
- make_current_frontier -- makes and saves a plot of the efficient frontier and the current price's relation to it
- make_isoforest_frontier -- makes and saves a plot of the efficient frontier, the current price's relation to it, and the suggested pricing for the products identified by the isolation forest
- make_full_frontier -- makes and saves a plot of the efficient frontier, the current price's relation to it, and the suggested pricing for the all the products
- make_isoforest_price_heatmap -- makes and saves a heatmap of the ensemble isolationforest prediction
- make_full_auto_price_var_heatmap -- makes and saves a heatmap of the fully automatic pricing solution

## efficient_frontier
This module contains functions relating to the efficient frontier calculations
- find_price_variants -- returns an array of prices to try, given the central price to edit around, the density of prices to try, and the maximum percentage to change the price by
- calc_pot_rev_profs -- returns an array of potential revenue and profits, given a base price, variations to attempt, beta, q, and cost
- calc_strat_weights -- returns weights to use for profit and revenue weighting given the number of strategies to attempt
- find_strategy_prof_rev -- finds profit revenue pairs which maximize the objective function, given the potential profits, potential revenues, profit weights, and revenue weights
- find_closest_strat -- returns a list of the prices which will optimize the current overall strategy
- calculate_q -- calculates the change in q from an old to new price, given q0, beta, old price, and new price
CurPriceStdVariation for the provided dataframe
- estimate_cur_strat -- calculates the angle that the summation of the current profit and revenue make in a plot of Profit vs revenue (used to pick where on the efiicient frontier to place prices)
- build_efficient_frontier -- returns three lists, with the revenue, profit, and all_prices, where the revenue and profit are the revenue and profit from different strategies, and all_prices is a list of the prices used to obtain each strategy
