'''This module contains helper functions related to building the efficient frontier,
both computationally and analytically'''
import numpy as np
import evaluation

def build_efficient_frontier(df):
    '''This function returns the revenue and profit of different strategies
    of the efficient frontier'''
    strategies = np.linspace(0, 1, 101)
    betas = df['FcstBeta']
    qs = df['Q']
    costs = df['Cost']
    rev = []
    prof = []
    all_prices = []
    for strategy in strategies:
        prices = strategy*costs + 1/betas
        revenue = evaluation.calculate_revenue(prices, betas, qs).sum()
        profit = evaluation.calculate_profit(prices, betas, qs, costs).sum()
        rev.append(revenue)
        prof.append(profit)
        all_prices.append(prices)
    return rev, prof, all_prices

def find_price_variants(price, density=201, max_change_percent=0.15):
    '''returns an array of prices to try, given the central price to edit around,
     the density of prices to try, and the maximum percentage to change the price by'''
    max_change = price*max_change_percent
    price_variations_to_try = np.linspace(-max_change, max_change, density)
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
    profit_weights = np.linspace(min_weight, max_weight, strategies)
    revenue_weights = np.linspace(max_weight, min_weight, strategies)
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
    for index in range(0, len(profit_weights)): #For each strategy
        profit_weight = profit_weights[index]
        revenue_weight = revenue_weights[index]
        #Index which maximizes strategy
        strat_ind = np.argmax(pot_profs*profit_weight + pot_revs*revenue_weight)
        profits.append(pot_profs[strat_ind])
        revenues.append(pot_revs[strat_ind])
    return np.array(profits), np.array(revenues)

def find_closest_strat(df, indexes=None):
    '''This function finds the best prices for all the indexes provided, with the
    objective of maximizing the distance the profit revenue graph goes away from
    the origin. Defaults to all indexes'''
    if indexes is None:
        indexes = df.index
    n_df = df.loc[indexes, :].copy()
    theta = estimate_cur_strat(df)
    rev, prof, all_prices = build_efficient_frontier(n_df)
    pot_thetas = np.arctan(np.array(prof) / np.array(rev))
    ind = np.argmin(abs(theta - pot_thetas))
    prices = all_prices[ind]
    return prices

def calculate_q(q0, beta, cur_price, new_price):
    '''This function calculates q at a new price'''
    Q = q0 * np.exp(-beta*(new_price-cur_price))
    return Q

def estimate_cur_strat(df):
    '''This function returns the given strategy, where the strategy is defined as
    the angle which forms a line out to the profit and revenue position a graph
    of profit vs revenue.'''
    strat_angle = np.arctan(df['CurProfit'].sum()/df['CurRev'].sum())
    return strat_angle
