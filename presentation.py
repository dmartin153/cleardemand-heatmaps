'''this module contains scripts used to achieve presentation results'''
import data_processing
import modeling
import evaluation
import confidential
import plotter
import heatmap
from bokeh.io import export_png
import numpy as np
import matplotlib.pyplot as plt

saveloc='figures/Presentation/'

def build_data_frame():
    '''This function builds the dataframe used for all the subsequent modeling'''
    fileloc = confidential.presentation_data_file()
    df = data_processing.main(fileloc)
    data_processing.add_price_variation(df)
    modeling.build_isoforest_preds(df)
    return df

def make_revenue_heatmap(df):
    '''This function builds the image for the heatmap based on revenue'''
    data = heatmap.HeatGrid(df=df,target='CurRev',normalization=0)
    app = heatmap.HeatMap(heatgrid=data,title='Revenue Heatmap')
    app.build()
    export_png(app.p, filename=saveloc+'CurrentRevenueHeatmap.png')

def make_price_heatmap(df):
    '''This function builds the image for the heatmap based on revenue'''
    data = heatmap.HeatGrid(df=df,target='CurPrice',normalization=0)
    app = heatmap.HeatMap(heatgrid=data,title='Price Heatmap')
    app.build()
    export_png(app.p, filename=saveloc+'CurrentPriceHeatmap.png')

def make_price_varstd_heatmap(df):
    '''This function builds the image for the heatmap based on revenue'''
    data = heatmap.HeatGrid(df=df,target='CurPriceStdVariation',normalization=0)
    app = heatmap.HeatMap(heatgrid=data,title='Price Variation Standard Deviation')
    app.build()
    export_png(app.p, filename=saveloc+'CurrentPriceStdHeatmap.png')

def make_isoforest_heatmap(df):
    '''This function builds the image for the heatmap isolation forest'''
    data = heatmap.HeatGrid(df=df,target='IsoForestPredict_ensemble',normalization=0)
    app = heatmap.HeatMap(heatgrid=data,title='Isolation Forest Ensemble Results')
    app.build()
    export_png(app.p, filename=saveloc+'IsoForestPredictEnsemble.png')

def make_q_and_errors(df):
    '''This function plots the Q vs price for the highest revenue product'''
    ind = np.argmax(df['CurRev'])
    fig = plotter.plot_q(df,ind)
    fig.savefig(saveloc+'Q_v_price.png')
    fig.close()

def make_product_prof_rev(df):
    '''This function plots the Revenue and Profit vs price for the highest revenue
    product'''
    ind = np.argmax(df['CurRev'])
    fig = plotter.dollar_v_price(df,ind)
    fig.savefig(saveloc+'Rev_prof_v_price.png')
    fig.close()

def make_product_frontier(df):
    '''this function plots the efficient frontier for a single product'''
    ind = np.argmax(df['CurRev'])
    fig = plotter.single_product_rev_v_profit(df,ind)
    fig.savefig(saveloc+'prof_v_rev.png')
    fig.close()

def make_current_frontier(df):
    '''This function plots the efficient frontier for an entier dataframe'''
    fig = plotter.efficient_frontier_plot(df)
    fig.savefig(saveloc+'efficient_frontier.png')
    fig.close()

def make_isoforest_frontier(df):
    '''This function plots the efficient frontier for the detected outliers'''
    ind = df[df['IsoForestPredict_ensemble']<6].index
    n_df = df.loc[ind,:].copy()
    fig = plotter.efficient_frontier_plot(n_df)
    rev = evaluation.calculate_revenue(n_df['IsoForestPrice'],n_df['FcstBeta'],n_df['Q'])
    prof = evaluation.calculate_profit(n_df['IsoForestPrice'],n_df['FcstBeta'],n_df['Q'],n_df['Cost'])
    plt.plot(rev.sum(),prof.sum(),'bo',label='Recommended Pricing')
    plt.legend()
    fig.savefig(saveloc+'identified_points_efficient_frontier.png')
    plt.close(fig)

def make_full_frontier(df):
    '''this function plots the efficient frontier, including all recommended prices'''
    fig = plotter.efficient_frontier_plot(df)
    rev = evaluation.calculate_revenue(df['FullAutoPricing'],df['FcstBeta'],df['Q'])
    prof = evaluation.calculate_profit(df['FullAutoPricing'],df['FcstBeta'],df['Q'],df['Cost'])
    plt.plot(rev.sum(),prof.sum(),'bo',label='Recommended Pricing')
    plt.legend()
    fig.savefig(saveloc+'full_efficient_frontier.png')
    plt.close(fig)

def make_isoforest_price_heatmap(df):
    '''This function builds the image for the heatmap isolation forest'''
    data = heatmap.HeatGrid(df=df,target='IsoForestPrice',sortby_y='IsoForestPrice',normalization=0)
    app = heatmap.HeatMap(heatgrid=data,title='Isolation Forest Ensemble Suggested Pricing')
    app.build()
    export_png(app.p, filename=saveloc+'IsoForestPredictEnsemblePrices.png')

def make_isoforest_price_var_heatmap(df):
    '''This function builds the image for the heatmap isolation forest'''
    data = heatmap.HeatGrid(df=df,target='IsoForestPrice',sortby_y='IsoForestPrice',normalization=1)
    app = heatmap.HeatMap(heatgrid=data,title='Isolation Forest Ensemble Suggested Pricing variation')
    app.build()
    export_png(app.p, filename=saveloc+'IsoForestPredictEnsemblePricesVar.png')

def make_full_auto_price_var_heatmap(df):
    '''This function builds the image for the heatmap isolation forest'''
    data = heatmap.HeatGrid(df=df,target='FullAutoPricing',sortby_y='FullAutoPricing',normalization=1)
    app = heatmap.HeatMap(heatgrid=data,title='Fully Automated Pricing Suggestion Variations')
    app.build()
    export_png(app.p, filename=saveloc+'FullAutoPricingHeatmap.png')
