'''This module contains functions used to evaluate prices'''
import numpy as np
def calculate_revenue(price, beta, q):
    '''This function calculates revenue given the  price, beta and q values.
    For CurRev Calculation, use price = CurPrice, beta = FcstBeta, and q = Q
    For RecRev calculation, use price = RecPrice, beta = FcstBeta, and q = Q'''
    unit_sales = q * np.exp(beta * - price)
    revenue = price * unit_sales
    return revenue

def calculate_profit(price, beta, q, cost):
    '''This function calculates revenue given the  price, beta, q, and cost values'''
    unit_sales = q * np.exp(beta * - price)
    profit = (price - cost) * unit_sales
    return profit

def rev_prof_gain(current_price, price, beta, q, cost):
    '''function which returns the change in revenue and profit from an old price
    to a new price, given the beta, q, and cost'''
    current_rev = calculate_revenue(current_price, beta, q)
    current_prof = calculate_profit(current_price, beta, q, cost)
    potential_rev = calculate_revenue(price, beta, q)
    potential_prof = calculate_profit(price, beta, q, cost)
    overall_rev_change = potential_rev - current_rev
    overall_prof_change = potential_prof - current_prof
    return overall_rev_change, overall_prof_change

def calculate_available_rev_prof(df):
    '''This function calculates the available profit for the difference between
    the recommendation and the current'''
    available_rev = 0.
    available_prof = 0.
    for _, row in df.iterrows():
        increased_revenue, increased_profit = rev_prof_gain(current_price=row['CurPrice'],
                                                            price=row['RecPrice'],
                                                            beta=row['FcstBeta'],
                                                            q=row['Q'],
                                                            cost=row['Cost'])
        available_rev += increased_revenue
        available_prof += increased_profit
    return available_rev, available_prof
