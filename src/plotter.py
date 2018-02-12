'''This module contains functions used for plotting'''
from __future__ import division
import itertools
import os
import evaluation
import clustering
import data_processing
import efficient_frontier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
    piv_tab = df.pivot_table(index=index, columns=column, values=value, aggfunc=np.mean)
    main_column = np.argmax(df.groupby(column)[value].var())
    main_index = np.argmax(df.groupby(index)[value].var())
    piv_tab.sort_values(by=main_column, axis=0, inplace=True, ascending=True, na_position='first')
    piv_tab.sort_values(by=main_index, axis=1, inplace=True, ascending=True, na_position='first')
    fig = plt.figure()
    ax = sns.heatmap(piv_tab)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(name.replace('_', ' '))
    fig.tight_layout()
    saveloc = 'figures/basic_heatmaps/'
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
    name = '{val}_heatmap__{ind}_vs_{col}_by_{sort}'.format(val=value, ind=index,
                                                            col=column, sort=sortby)
    piv_tab = df.pivot_table(index=index, columns=column, values=[value, sortby], aggfunc=np.mean)
    main_column = np.argmax(df.groupby(column)[value].var())
    main_index = np.argmax(df.groupby(index)[value].var())
    piv_tab.sort_values(by=(sortby, main_column), axis=0, inplace=True,
                        ascending=True, na_position='first')
    piv_tab.sort_values(by=main_index, axis=1, inplace=True, ascending=True, na_position='first')
    fig = plt.figure()
    ax = sns.heatmap(piv_tab[value])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(name.replace('_', ' '))
    fig.tight_layout()
    saveloc = 'figures/sorted_heatmaps/'
    check_dir(saveloc)
    fig.savefig(saveloc+name+'.jpg')
    plt.close(fig)

def correlation_plots(df, x_col_name, y_col_name):
    '''This function makes scatter plots between the given columns, and provides
    correlation values'''
    n_df = drop_nans(df, [x_col_name, y_col_name])
    num_remaining_entries = len(n_df[x_col_name])
    if num_remaining_entries == 0:
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
    plt.plot(x, y, '.', label='Correlation = {}'.format(corr), alpha=0.1)
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

def plot_ppf(df, title=None):
    '''this function plots a basic production possibility frontier'''
    sns.set(font_scale=1.6, rc={'lines.linewidth': 2.5})
    profits = clustering.find_labeled_points('KeyProfitPoints', df)
    revs = clustering.find_labeled_points('KeyRevPoints', df)
    fig, ax = plt.subplots()
    plt.plot(revs, profits, 'g.', alpha=0.5, label='Strategy Variants')
    plt.plot(df['CurRev'].sum(), df['CurProfit'].sum(), 'bo', label='Current Prices')
    plt.plot(df['RecRev'].sum(), df['RecProfit'].sum(), 'ko', label='Recommended Prices')
    ef_revs, ef_profs, _ = efficient_frontier.build_efficient_frontier(df)
    plt.plot(ef_revs, ef_profs, 'r', label='Efficient Frontier')
    plt.xlabel('Total Revenue')
    plt.ylabel('Total Profit')
    plt.legend()
    if title:
        plt.title(title)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    plt.tight_layout()
    return fig

def cluster_fake_ppf(df, columns=None, n_clusters=2, title=None):
    '''Clusters along the indicated column into the n_clusters provided, and
    returns a figure of the computationally computed efficient frontier'''
    if columns is None:
        columns = ['CurRev']
    kmeans = KMeans(n_clusters=n_clusters, random_state=20)
    X = np.array(df[columns])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kmeans.fit(X)
    df['cluster_labels'] = kmeans.labels_
    fig = plot_ppf(df, title)
    return fig

def make_and_save_ppf(df, name, cluster_cols=None, n_clusters=3, n_strats=8):
    '''performs all the computation required to calculate the computational
    efficient frontier and makes and saves a figure with the provided inputs'''
    if cluster_cols is None:
        cluster_cols = ['CurRev']
    saveloc = '../figures/PPFs/'
    check_dir(saveloc)
    plot_name = name+'_{}strats_{}clusters_of_{}'.format(n_strats, n_clusters,
                                                         '_'.join(cluster_cols))
    n_df = df.copy()
    data_processing.add_key_points(n_df, strategies=n_strats)
    fig = cluster_fake_ppf(n_df, columns=cluster_cols, n_clusters=n_clusters, title=plot_name)
    fig.savefig(saveloc+plot_name+'.jpg')
    plt.close(fig)

def find_rev_and_prof(df, index):
    '''This function returns the revenue and profit and their bounds for the given index of
    the provided dataframe'''
    current_price = df['CurPrice'][index]
    beta = df['FcstBeta'][index]
    q = df['Q'][index]
    cost = df['Cost'][index]
    price_variants = efficient_frontier.find_price_variants(current_price, max_change_percent=1.)
    prices = price_variants+current_price
    beta_max = 1/df['AlphaMin'][index]
    beta_min = 1/df['AlphaMax'][index]
    quantities, quant_upper, quant_lower = efficient_frontier.calculate_quantity(q, beta, current_price,
                                                                                 prices, beta_min, beta_max)
    revenue = quantities * prices
    profit = quantities * (prices - cost)
    upper_rev = quant_upper * prices
    lower_rev = quant_lower * prices
    upper_prof = quant_upper * (prices - cost)
    lower_prof = quant_lower * (prices - cost)
    return prices, revenue, upper_rev, lower_rev, profit, upper_prof, lower_prof

def dollar_v_price(df, index):
    '''This function makes a plot of dollars vs price for the provided index into the
    data frame'''
    prices, revenue, upper_rev, lower_rev, profit, upper_prof, lower_prof = find_rev_and_prof(df, index)
    sns.set(font_scale=1.6, rc={'lines.linewidth': 2.5})
    fig, ax = plt.subplots()
    plt.plot(prices, revenue, 'r', label='Revenue')
    # plt.plot(prices, upper_rev, 'r--', label='Revenue Bounds')
    # plt.plot(prices, lower_rev, 'r--')
    plt.plot(prices, profit, 'g', label='Profit')
    plt.axvline(x=df['CurPrice'][index], color='b', linestyle='--', label='Current Price')

    # plt.plot(prices, upper_prof, 'b--', label='Profit Bounds')
    # plt.plot(prices, lower_prof, 'b--')
    plt.xlabel('Price ($)')
    plt.ylabel('Return ($)')
    plt.title('Revenue & Profit vs price')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    plt.legend()
    plt.tight_layout()
    return fig

def single_product_rev_v_profit(df, index, bounds=0):
    '''This function makes a plot of revenue vs profit for the provided index for a variety of
    prices'''
    prices, revenue, upper_rev, lower_rev, profit, upper_prof, lower_prof = find_rev_and_prof(df, index)
    # current_price = df['CurPrice'][index]
    # beta = df['FcstBeta'][index]
    # q = df['Q'][index]
    # cost = df['Cost'][index]
    # price_variants = efficient_frontier.find_price_variants(current_price, max_change_percent=0.15)
    # prices = price_variants+current_price
    # if bounds == 1:
    #     beta_max = 1. / df['AlphaMin'][index]
    #     q_max = df['CurQty'][index] / np.exp(-beta_max * df['CurPrice'][index])
    #     beta_min = 1. / df['AlphaMax'][index]
    #     q_min = df['CurQty'][index] / np.exp(-beta_min * df['CurPrice'][index])
    #     max_beta_rev = evaluation.calculate_revenue(prices, beta_max, q_max)
    #     min_beta_rev = evaluation.calculate_revenue(prices, beta_min, q_min)
    #     max_beta_prof = evaluation.calculate_profit(prices, beta_max, q_max, cost)
    #     min_beta_prof = evaluation.calculate_profit(prices, beta_min, q_min, cost)
    # revenue = evaluation.calculate_revenue(prices, beta, q)
    # profit = evaluation.calculate_profit(prices, beta, q, cost)
    sns.set(font_scale=1.6, rc={'lines.linewidth': 2.5})
    fig, ax = plt.subplots()
    plt.plot(revenue, profit, 'r', label='Price Frontier')
    # if bounds == 1:
    # plt.plot(upper_rev, upper_prof, 'r--', label='Price Frontier Error Bounds')
    # plt.plot(lower_rev, lower_prof, 'r--')
    plt.plot(df.loc[index, 'CurRev'], df.loc[index, 'CurProfit'], 'ob', label='Current Price Point')
    plt.xlabel('Revenue ($)')
    plt.ylabel('Profit ($)')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    plt.title('Revenue Profit Curve')
    plt.legend()
    plt.tight_layout()
    return fig

def find_rev_and_prof_ef(df):
    '''This function finds the maximum and minimum profit and revenue for the efficient
    frontier of the provided dataframe'''
    rev, prof, all_prices = efficient_frontier.build_efficient_frontier(df)
    rev_min = []
    rev_max = []
    prof_min = []
    prof_max = []
    for prices in all_prices:
        max_beta = 1 / df['AlphaMax']
        min_beta = 1 / df['AlphaMin']
        max_q = df['Q']*np.exp(-df['FcstBeta']*df['CurPrice'])/np.exp(-max_beta*df['CurPrice'])
        min_q = df['Q']*np.exp(-df['FcstBeta']*df['CurPrice'])/np.exp(-min_beta*df['CurPrice'])
        rev_min.append(evaluation.calculate_revenue(prices, min_beta, min_q).sum())
        rev_max.append(evaluation.calculate_revenue(prices, max_beta, max_q).sum())
        prof_min.append(evaluation.calculate_profit(prices, min_beta, min_q, df['Cost']).sum())
        prof_max.append(evaluation.calculate_profit(prices, max_beta, max_q, df['Cost']).sum())
    return rev, prof, rev_min, prof_min, rev_max, prof_max

def efficient_frontier_plot(df):
    '''This function makes a plot of maximum revenue vs profit for the provided dataframe'''
    rev, prof, _, _, _, _ = find_rev_and_prof_ef(df)
    sns.set(font_scale=1.6, rc={'lines.linewidth': 2.5})
    fig, ax = plt.subplots()
    plt.plot(rev, prof, 'r', label='Efficient Frontier')
    plt.plot(df['CurRev'].sum(), df['CurProfit'].sum(), 'ko', label='Current Position')
    plt.plot(rev[0], prof[0], 'ro', label='Maximize Revenue')
    plt.plot(rev[-1], prof[-1], 'go', label='Maximize Profit')
    plt.xlabel('Revenue ($)')
    plt.ylabel('Profit ($)')
    plt.title('Profit Vs Revenue')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_q(df, index):
    '''This function plots Q vs price for a variety of prices for the dataframe's indexed row'''
    current_price = df['CurPrice'][index]
    beta = df['FcstBeta'][index]
    q = df['Q'][index]
    price_variants = efficient_frontier.find_price_variants(current_price, max_change_percent=1.0)
    prices = price_variants+current_price
    sns.set()
    fig = plt.figure()
    Qs = efficient_frontier.calculate_q(q, beta, current_price, prices)
    beta_max = 1/df['AlphaMin'][index]
    beta_min = 1/df['AlphaMax'][index]
    beta_sigma = (beta_max - beta_min) / 2
    perc_beta_sigma = beta_sigma/beta
    perc_sigma = perc_beta_sigma*beta*(prices-current_price)
    qupper = Qs + (Qs*np.exp(perc_sigma) - Qs)
    qlower = Qs - (Qs*np.exp(perc_sigma) - Qs)
    plt.plot(prices, Qs, 'k', label='Q')
    plt.plot(prices, qupper, 'k--', label='Q Bounds')
    plt.plot(prices, qlower, 'k--')
    plt.xlabel('Price ($)')
    plt.ylabel('Quantity (#)')
    plt.title('Quantity vs price')
    plt.legend()
    plt.tight_layout()
    return fig

def plot_quantity(df, index):
    '''This function plots Q vs price for a variety of prices for the dataframe's indexed row'''
    current_price = df['CurPrice'][index]
    beta = df['FcstBeta'][index]
    q = df['Q'][index]
    price_variants = efficient_frontier.find_price_variants(current_price, max_change_percent=1.)
    prices = price_variants+current_price
    sns.set(font_scale=1.6, rc={'lines.linewidth': 2.5})
    fig, ax = plt.subplots()
    beta_max = 1/df['AlphaMin'][index]
    beta_min = 1/df['AlphaMax'][index]
    quantities, quant_upper, quant_lower = efficient_frontier.calculate_quantity(q, beta, current_price,
                                                                                 prices, beta_min, beta_max)
    plt.plot(prices, quantities, 'k', label='Quantity')
    plt.axvline(x=current_price, color='b', linestyle='--', label='Current Price')
    # plt.plot(prices, quant_upper, 'k--', label='Q Bounds')
    # plt.plot(prices, quant_lower, 'k--')
    plt.xlabel('Price ($)')
    plt.ylabel('Quantity (#)')
    plt.title('Quantity vs price')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    plt.legend()
    plt.tight_layout()
    return fig
