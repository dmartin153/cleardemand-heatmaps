'''This module contains functions used for data processing'''
import pandas as pd
import pdb
import numpy as np
import confidential
import evaluation
import efficient_frontier
import modeling

def small_load(fileloc):
    '''this function takes a file location as input and outputs a dataframe with
    basic information.
    Inputs:
    fileloc -- string with relative location of data file with csv
    Outputs:
    df -- pandas dataframe with simplified columns
    By: David Martin
    On: Jan 9, 2018'''
    cols = confidential.small_load_columns()
    df = pd.read_csv(fileloc, usecols=cols)
    convert_dol_to_num(df)
    return df

def convert_dol_to_num(df):
    '''This function columns of a dataframe which have dollar amounts from a
    string to a float.
    Inputs:
    df -- dataframe from the data
    Outputs:
    none
    By: David Martin
    On: Jan 10, 2018'''
    for col in df.columns:
        df[col] = df[col].apply(fix_dol)

def fix_dol(x):
    '''This function takes removes the $ and returns a string, should the given
    input be a string with a beginning $
    Inputs:
    x -- object to check and potentially remove a $ from
    Returns:
    if x was string with $ beginning -- float(x[1:])
    else x
    By: David Martin
    On: Jan 10, 2018'''
    if type(x) == type('str'):
        if x[0] == '$':
            return float(x[1:])
    return x

def main(fileloc=None):
    '''This process builds the general dataframe for use in other modules'''
    if fileloc == None:
        fileloc='PriceBook.csv'
    df = pd.read_csv(fileloc)
    convert_dol_to_num(df)
    drop_rows(df)
    if not 'Q' in df.columns:
        add_q(df)
    add_price_variation(df)
    # modeling.add_iso_sug_price(df)
    return df

def drop_rows(df):
    '''this function drops products which are not offered in all areas'''
    drop_prods = []
    prods = df['ProductId'].unique()
    for prod in prods:
        if sum(df['ProductId']==prod) < max(df['ProductId'].value_counts()):
            drop_prods.append(prod)
    for drop_prod in drop_prods:
        df.drop(df[df['ProductId'] == drop_prod].index, inplace=True)

def build_num_prices(df):
    '''this function adds columns of "NumIdenticalPrices" and "NumTotalProducts."
    NumIdenticalPrices -- the number of identical prices that product has
    NumTotalProducts -- the number of this product available in the database'''
    for product in df['ProductId'].unique():
        prices_series = df[df['ProductId']==product]['CurPrice'].value_counts()
        for key in prices_series.keys():
            indicies = df[(df['ProductId'] == product) & (df['CurPrice'] == key)].index
            df.loc[indicies,'NumIdenticalPrices'] = prices_series[key]
            df.loc[indicies,'NumTotalProducts'] = sum(prices_series)

def build_success_metrics(df):
    '''this function adds columns of "NormAreaProfit", "NormAreaRev", "Strategy",
    "RegretRev", "RegretProfit", "NormProductProfit", ad "NormProductRev'"
    "NormAreaProfit" -- number of standard deviationns from the average profit in that area
    "NormAreaRev" -- number of standard deviations from the average revenue in that area
    "NormProductProfit" -- number of standard deviations from the average profit for that product
    "NormProductRev" -- number of standard deviations from the average revenue for that product
    "CurStrategy" -- CurProfit / CurRevenue
    "RecStrategy" -- RecProfit / RecRevenue
    "RegretRev" -- Difference  between recommended revenue and current revenue
    "RegretProfit" -- Difference between Recommended Profit and current profit'''
    metrics  = ['Profit', 'Rev']
    norm_tos = ['Product', 'Area']
    for metric in metrics:
        for norm_to in norm_tos:
            for product in df['{}Id'.format(norm_to)].unique():
                inds = df[df['{}Id'.format(norm_to)]==product].index
                target = 'Cur{}'.format(metric)
                avg = df.loc[inds,target].mean()
                std = df.loc[inds,target].std()
                df.loc[inds,'Norm{}{}'.format(norm_to,metric)] = (df.loc[inds,target] - avg) / std
    df['CurStrategy'] = df['CurProfit'] / df['CurRev']
    df['RecStrategy'] = df['RecProfit'] / df['RecRev']
    df['RegretRev'] = df['CurRev'] - df['RecRev']
    df['RegretProfit'] = df['CurProfit'] - df['RecProfit']

def add_q(df):
    '''This function adds Q to the dataframe, requires CurPrice, FcstBeta, and CurRev'''
    df['Q'] = df['CurRev'] / (df['CurPrice'] * np.exp(-df['CurPrice'] * df['FcstBeta']))

def add_key_points(df,strategies=10):
    '''This function builds the key profit and revenue points for different strategies
    for a dataframe. Used to plot the production possibility frontier of the dataset.'''
    prof_points = []
    rev_points = []
    for index, row in df.iterrows():
        price_variations_to_try = efficient_frontier.find_price_variants(row['CurPrice'])
        pot_revs, pot_profs = efficient_frontier.calc_pot_rev_profs(row['CurPrice'], price_variations_to_try, row['FcstBeta'], row['Q'], row['Cost'])
        profit_weights, revenue_weights = efficient_frontier.calc_strat_weights(strategies)
        strat_profits, strat_revs = efficient_frontier.find_strategy_prof_rev(pot_revs, pot_profs, profit_weights, revenue_weights)
        prof_points.append(strat_profits)
        rev_points.append(strat_revs)
    df['KeyProfitPoints'] = prof_points
    df['KeyRevPoints'] = rev_points

def add_price_variation(df):
    '''this function adds a column "CurPriceVariation" which is the variation of
    the CurPrice from the average CurPrice for that product'''
    avgs = dict()
    stds = dict()
    xs = df['ProductId'].unique()
    for x in xs:
        avgs[x] = df[df['ProductId'] == x]['CurPrice'].mean()
        stds[x] = df[df['ProductId'] == x]['CurPrice'].std()
    raw_variation = df['CurPrice'].values - np.array([avgs[key] for key in df['ProductId'].values])
    standardized_variation = raw_variation / np.array([stds[key] for key in df['ProductId'].values])
    df['CurPriceVariation'] = raw_variation
    df['CurPriceStdVariation'] = np.nan_to_num(standardized_variation)

def build_isoforest_preds(df):
    '''This function adds isolation forest predictions for the given data set. Specifically,
    it builds seven models, they are divided as follows:
    The following are trained off the entire dataset:
    IsoForestPredict_std -- prediction made from an isolation forest on the standard deviations
    IsoForestPredict_var -- prediction made from an isolation forest on the absolute variation
    IsoForestPredict_stdvar -- prediction made from an isolation forest on the standard deviation and absolute variations
    The Following are trained off only the top 256 revenue items:
    IsoForestPredict_std_toprev -- prediction made from an isolation forest on the standard deviation
    IsoForestPredict_var_toprev -- prediction made from an isolation forest on the absolute variation
    IsoForestPredict_stdvar_toprev -- prediction made from an isolation forest on the standard deviation and absolute variation

    IsoForestPredict_ensemble -- voting ensemble of the above six models'''
    fit_options = ['all', 'toprev']
    training_options = ['std', 'var', 'stdvar']
    df['IsoForestPredict_ensemble'] = 0
    for fit_option in fit_options:
        for training_option in training_options:
            preds = modeling.isoforestpred(df,fit_option,training_option)
            df['IsoForestPredict_'+fit_option+'_'+training_option] = preds
            df['IsoForestPredict_ensemble'] += preds
    df['IsoForestPrice'] = df['CurPrice']
    inds = df['IsoForestPredict_ensemble']<max(df['IsoForestPredict_ensemble'])
    prices = plotter.find_closest_strat(df,indexes=inds)
    df.loc[inds, 'IsoForestPrice'] = prices.values
    all_prices = plotter.find_closest_strat(df)
    df['FullAutoPricing'] = all_prices
