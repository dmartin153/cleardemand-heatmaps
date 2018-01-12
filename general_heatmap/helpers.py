'''This module contains helper functions for the heatmap application'''
from bokeh.layouts import column
from bokeh.plotting import figure, curdoc
import logging
logging.basicConfig()
from bokeh.io import show, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper,
    LinearColorMapper,
    BasicTicker,
    CategoricalTicker,
    PrintfTickFormatter,
    ColorBar,
    LogTicker
)
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
import data_processing
from math import pi
import pdb
import confidential
import helpers
import numpy as np


def build_source(x_col='default',y_col='default', value='default',
    sort_columns=['default'],x_name='default',y_name='default', division='default',
    mini='-inf', maxi='inf', diff='default'):
    '''this builds the source for a heatmap'''
    #build Defaults
    if x_col == 'default':
        x_col = confidential.x_col()
    if y_col == 'default':
        y_col = confidential.y_col()
    if value == 'default':
        value = confidential.value()
    if sort_columns == ['default']:
        sort_columns = confidential.sort_columns()
    if x_name == 'default':
        x_name = confidential.x_name()
    if y_name == 'default':
        y_name = confidential.y_name()
    if division == 'default':
        division = confidential.division()
    if diff == 'default':
        diff = confidential.difference()

    df = clean_df(sort_columns, division, value=value, mini=mini, maxi=maxi)
    raw_xs = df[x_col].unique()
    raw_ys = df[y_col].unique()
    interpret_xs = [(ind, raw_x) for ind,raw_x in enumerate(raw_xs)]
    interpret_ys = [(ind, raw_y) for ind,raw_y in enumerate(raw_ys)]

    x_locs = []
    for x_val in df[x_col]:
        for interpret_x in interpret_xs:
            if interpret_x[1] == x_val:
                x_locs.append(interpret_x[0])
    y_locs = []
    for y_val in df[y_col]:
        for interpret_y in interpret_ys:
            if interpret_y[1] == y_val:
                y_locs.append(interpret_y[0])

    y_names = df[y_name].values
    x_names = df[x_name].values

    target = df[value].values
    normalization = df[diff].values

    source = ColumnDataSource(data=dict(
        x=x_locs,
        y=y_locs,
        x_name=x_names,
        y_name=y_names,
        val=target - normalization,
    ))
    return source

def build_mapper(source, scale=0):
    '''This function builds a mapper and color_bar for a graph given the source'''
    pal='Plasma256'
    if scale==0:
        mapper = LinearColorMapper(palette=pal, low=min(source.data['val']), high=max(source.data['val']))
        color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(),
                            label_standoff=12, border_line_color=None, location = (0,0))
    elif scale==1:
        mapper = LogColorMapper(palette=pal, low=min(source.data['val']), high=max(source.data['val']))
        color_bar = ColorBar(color_mapper=mapper, ticker=LogTicker(),
                            label_standoff=12, border_line_color=None, location = (0,0))
    return mapper, color_bar

def clean_df(sort_columns, division, value='default', mini='-inf', maxi='inf'):
    '''This function builds a pandas dataframe with the division and sorting provided'''
    df = data_processing.main()
    df = df[df[division[0]] == division[1]]
    df.sort_values(by=sort_columns, inplace=True)
    if value == 'default':
        value = confidential.value()
    mini = np.float(mini)
    maxi = np.float(maxi)
    df.drop(df[df[value] < mini].index,inplace = True)
    df.drop(df[df[value] > maxi].index,inplace = True)
    return df

def general_heatmap(source,x_col='default',y_col='default', value='default',
        sort_columns=['default'],x_name='default',y_name='default', mini='-inf',
        maxi='inf'):
    '''this function builds and returns a Bokeh figure of the heatmap'''
    if sort_columns == ['default']:
        sort_columns = confidential.sort_columns()
    division = confidential.division()
    df = clean_df(sort_columns, division, value=value, mini=mini, maxi=maxi)
    TOOLS = "pan,wheel_zoom,reset,hover,save"

    p = figure(
        title="Interactive Heatmap", tools=TOOLS,
        x_axis_location='above',
        toolbar_location='below'
    )
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = '12pt'
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    mapper, color_bar = build_mapper(source)
    p.rect(x='x', y='y', width=1, height=1,
            source=source,
            fill_color={'field': 'val', 'transform': mapper},
            line_color=None)

    build_hover(p, source)

    return p, mapper

def build_hover(p,source, x_name='default', y_name='default', value='default'):
    '''This function updates the hovertool of the figure provided'''
    if x_name == 'default':
        x_name = confidential.x_name()
    if y_name == 'default':
        y_name = confidential.y_name()
    if value == 'default':
        value = confidential.value()
    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        (x_name, "@x_name"),
        (y_name, "@y_name"),
        (value, "@val"),
    ]
