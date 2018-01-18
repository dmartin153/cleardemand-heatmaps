'''This module contains functions used for modeling'''
import numpy as np
import pandas as pd
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
