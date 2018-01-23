'''This module contains functions used for modeling'''
import numpy as np
import pandas as pd
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import evaluation

def feature_cleaning(df,cols,target):
    '''This function is used to divide the provided data into an X feature matrix
    and a y target vector'''
    df = df.dropna(subset=[[target] + cols]) # Drop nans involved in the features and targets
    y = df[target].values
    df_norm = (df[cols] - df[cols].mean()) / (df[cols].max() - df[cols].min())
    X = np.insert(df_norm.values, 0, 1, axis=1)
    return X, y

def build_regression_model(df,model=LinearRegression(),features=['NumIdenticalPrices','Cost','CurPrice'],
            target='CurRev', rs=20):
    '''This function performs a train test split and applies the provided sklearn
    regression model to the dataframe provided.'''
    X, y = feature_cleaning(df,cols=features, target=target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rs)
    model.fit(X_train,y_train)
    return model, X_train, y_train, X_test, y_test

def iso_forest_predict_outliers(df):
    '''This function predicts outliers using an isolation forest and the CurPriceStdVariation'''
    iso_forest = IsolationForest(n_jobs=-1, random_state=22,contamination=0.0001)
    outliers = []
    for prod in df['ProductId'].unique(): # for each product
        selection_matrix = df['ProductId'] == prod
        X = df[selection_matrix]['CurPriceStdVariation'].values
        weights = df[selection_matrix]['CurRev'].values
        X = X.reshape(-1, 1)
        iso_forest.fit(X,sample_weight=weights)
        preds = iso_forest.predict(X)
        outlier_inds = df[df['ProductId'] == prod][preds==-1].index
        outliers.append(outlier_inds)
    return outliers

def estimate_cur_strat(df):
    '''This function returns the given strategy, where the strategy is defined as
    the angle which forms a line out to the profit and revenue position a graph
    of profit vs revenue.'''
    strat_angle = np.arctan(df['CurProfit'].sum()/df['CurRev'].sum())
    return strat_angle

def build_efficient_frontier(df):
    '''This function returns the revenue and profit of different strategies of the efficient frontier'''
    strategies = np.linspace(0,1,101)
    betas = df['FcstBeta']
    qs = df['Q']
    costs = df['Cost']
    rev = []
    prof = []
    all_prices = []
    for strategy in strategies:
        prices = strategy*costs + 1/betas
        revenue = evaluation.calculate_revenue(prices,betas,qs).sum()
        profit = evaluation.calculate_profit(prices,betas,qs,costs).sum()
        rev.append(revenue)
        prof.append(profit)
        all_prices.append(prices)
    return rev, prof, all_prices

def build_new_suggested_prices(df):
    '''This function creates a new column of IsoSugPrice, which is based off the isolation
    forest ensemble column'''
    dct = {}
    df['IsoSugPrice'] = df['CurPrice'] #Initialize as current price
    for outlier in outliers:
        df.loc[outlier, 'IsoSugPrice'] = df.loc[outlier, 'IsoSugPrice'] - df.loc[outlier, 'CurPriceVariation'] / 2

def add_iso_sug_price(df):
    outliers = iso_forest_predict_outliers(df)
    build_new_suggested_prices(df,outliers)

def isoforestpred(df,fit_option, training_option):
    n_df = df.copy()
    n_df = df.sort_values(by='CurRev')
    if training_option == 'std':
        features = 'CurPriceStdVariation'
    elif training_option == 'var':
        features = 'CurPriceVariation'
    elif training_option == 'stdvar':
        features = ['CurPriceStdVariation','CurPriceVariation']
    if fit_option == 'toprev': #If selecting only top 256 values
        sample_limit = 256
    else:
        sample_limit = len(n_df)
    X = n_df.loc[:n_df.index[sample_limit-1],features].values
    full_data = df[features].values
    if training_option != 'stdvar':
        X = X.reshape(-1,1)
        full_data = full_data.reshape(-1,1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    iso_for = IsolationForest(contamination = 0.01, random_state=42,max_samples=sample_limit)
    iso_for.fit(X)
    full_data = scaler.fit_transform(full_data)
    preds = iso_for.predict(full_data)
    return preds

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
            preds = isoforestpred(df,fit_option,training_option)
            df['IsoForestPredict_'+fit_option+'_'+training_option] = preds
            df['IsoForestPredict_ensemble'] += preds
