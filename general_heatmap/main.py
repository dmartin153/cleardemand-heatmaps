'''This has the main execution of the heatmap app'''
from bokeh.layouts import column
from bokeh.plotting import figure, curdoc
import logging
logging.basicConfig()
from bokeh.io import output_file
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
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, layout
from bokeh.models.widgets import Select
import data_processing
from math import pi
import pdb
import confidential
import helpers

#Build the source data
source = helpers.build_source()
#User the source data to build a figure and color map
p, mapper = helpers.general_heatmap(source)
#Get a color bar to interpret the figure
_, color_bar = helpers.build_mapper(source)
#Add the color bar to the figure
p.add_layout(color_bar, 'right')
#Instantiate a dataframe to reference columns
df = data_processing.main()
cols = list(df.select_dtypes([int,float]).columns)
init_val = confidential.value()
default_x = confidential.x_col()
default_y = confidential.y_col()

def update():
    '''This method updates the figure based on user inputs to the controls'''
    #Build a new source
    n_source = helpers.build_source(x_col=x_axis.value, y_col=y_axis.value, sort_columns=[sorted_by.value], value=target.value)
    #Build a new mapper
    n_mapper,_ = helpers.build_mapper(n_source)
    #Update the old source with the new source's data
    source.data = n_source.data
    #Update the high and low points on the mapper and color bar
    color_bar.color_mapper.high = n_mapper.high
    color_bar.color_mapper.low = n_mapper.low
    mapper.high = n_mapper.high
    mapper.low = n_mapper.low
    #Update the hovertool text
    helpers.build_hover(p, source, value=target.value)

###Make the controls
#Column to sort the data by
sorted_by = Select(title="Sort By", options=cols, value=init_val)
#Column to use as a target for plotting
target = Select(title='Target', options=cols, value=init_val)
#Colun to use to set x axis
x_axis = Select(title='X axis', options=cols, value=default_x)
#Column to use to set y axis
y_axis = Select(title='Y axis', options=cols, value=default_y)


#Set how controls impact changes
controls = [sorted_by, target, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

### Build the final displace
inputs = widgetbox(*controls)
l = layout([
    [p, inputs]
])

#Do initial update of values
update()

#Set up document
curdoc().add_root(l)
curdoc().title = 'General Heatmap'
