'''This module contains functions used for data processing'''
import pandas as pd
import pdb
import confidential

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
        fileloc=confidential.filelocation()
    df = pd.read_csv(fileloc)
    convert_dol_to_num(df)
    return df
