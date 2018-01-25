'''This module contains functions used for modeling'''
import numpy as np
import pandas as pd
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import evaluation
import plotter

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

def build_new_suggested_prices(df,outliers):
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
    '''returns predicted outliers for the dataframe provided with the fit and training options provided'''
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
