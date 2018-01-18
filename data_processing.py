'''This module contains functions used for data processing'''
import pandas as pd
import pdb
import numpy as np
import confidential
import evaluation

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

def pot_revenue_profit(df,density=201,max_change=1):
    price_variations_to_try = np.linspace(-max_change,max_change,density)
    return price_variations_to_try

def calc_pot_rev_profs(base_price, variations, beta, q, cost):
    '''This function calculates the potential revenues and profits for the provided
    variations to the given price
    Inputs:
        base_price - float, price to base variations off of
        variations - numpy array of variations to apply to price
        beta - float, beta value for elasticity
        q - float, q value for elasticity
        cost - float, cost of product
    Outputs:
        pot_revs - numpy array with revenues for each variation
        pot_prof - numpy array with profit for each variation '''
    pot_revs = []
    pot_profs = []
    for price_variation in variations:
        price = base_price + price_variation
        rev = evaluation.calculate_revenue(price, beta, q)
        prof = evaluation.calculate_profit(price, beta, q, cost)
        pot_revs.append(rev)
        pot_profs.append(prof)
    return np.array(pot_revs), np.array(pot_profs)

def calc_strat_weights(strategies=4):
    '''This calculates the weights to give each strategy
    Inputs:
        strategies - int, number of strategies to consider
    Outputs:
        profit_weights - numpy array, amount to weight each profit for the given strategy
        revenue_weights - numpy array, amount to oweight each revenue for the given strategy '''
    min_weight = 0
    max_weight = 1
    profit_weights = np.linspace(min_weight,max_weight,strategies)
    revenue_weights = np.linspace(max_weight,min_weight,strategies)
    return profit_weights, revenue_weights

def find_strategy_prof_rev(pot_revs, pot_profs, profit_weights, revenue_weights):
    '''This function finds profit revenue pairs which maximize returns for the given weights
    Inputs:
        pot_revs - numpy array, revenues for different pricings
        pot_profs - numpy array, profits for different pricings
        profit_weights - numpy array, profit weight for each strategy
        revenue_weights - numpy array, revenue weight for each strategy
    Outputs:
        profits - profits for each strategy
        revenues - revenues for each strategy'''
    profits = []
    revenues = []
    for index in range(0,len(profit_weights)): #For each strategy
        profit_weight = profit_weights[index]
        revenue_weight = revenue_weights[index]
        strat_ind = np.argmax(pot_profs*profit_weight + pot_revs*revenue_weight) #Index which maximizes strategy
        profits.append(pot_profs[strat_ind])
        revenues.append(pot_revs[strat_ind])
    return np.array(profits), np.array(revenues)

def add_key_points(df,strategies=4):
    '''This function builds the production possibility frontier for the given data frame'''
    price_variations_to_try = np.arange(-1., 1.01, 0.01)
    prof_points = []
    rev_points = []
    for index, row in df.iterrows():
        pot_revs, pot_profs = calc_pot_rev_profs(row['CurPrice'], price_variations_to_try, row['FcstBeta'], row['Q'], row['Cost'])
        profit_weights, revenue_weights = calc_strat_weights(strategies)
        strat_profits, strat_revs = find_strategy_prof_rev(pot_revs, pot_profs, profit_weights, revenue_weights)
        prof_points.append(strat_profits)
        rev_points.append(strat_revs)
    df['KeyProfitPoints'] = prof_points
    df['KeyRevPoints'] = rev_points
