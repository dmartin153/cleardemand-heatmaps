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
from bokeh.models.widgets import Select, TextInput, RadioGroup
import data_processing
from math import pi
import pdb
import confidential
import heatmap

#Build the source data
app = heatmap.Heatmap()
# app.grab_data()
# app.selection_data()
# app.build_axes_interpreters()
# app.generate_source()
# app.generate_figure()
# app.hovertool()
app.build()

app.p.add_layout(app.color_bar, 'right')
#Instantiate a dataframe to reference columns
df = data_processing.main()
cols = list(df.select_dtypes([int,float]).columns)
#Load defaults for selection menues
init_val = confidential.value()
default_x = confidential.x_col()
default_y = confidential.y_col()
default_div = confidential.division()
init_diff = confidential.difference()

def update():
    '''This method updates the figure based on user inputs to the controls'''
    #Build a new source
    app.x_axis=x_axis.value
    app.y_axis=y_axis.value
    app.sortby = sorted_by.value
    app.target = target.value
    app.min_tar_perc = mini_val.value
    app.max_tar_perc = maxi_val.value
    app.selection_criteria = (division_cat.value, int(division_val.value))


    # app.generate_source(x_axis=x_axis.value, y_axis=y_axis.value,
                        # sort_columns=[sorted_by.value], value=target.value,
                        # division=(division_cat.value, int(division_val.value)),
                        # mini=mini_val.value, maxi=maxi_val.value,
                        # diff=difference.value)
    #Build a new mapper
    app.update()
    # app.generate_source()
    # app.hovertool()
    # n_mapper,_ = helpers.build_mapper(n_source)
    # #Update the old source with the new source's data
    # source.data = n_source.data
    # #Update the high and low points on the mapper and color bar
    # color_bar.color_mapper.high = n_mapper.high
    # color_bar.color_mapper.low = n_mapper.low
    # mapper.high = n_mapper.high
    # mapper.low = n_mapper.low
    # #Update the hovertool text
    # helpers.build_hover(p, source, value=target.value)

###Make the controls
#Column to sort the data by
sorted_by = Select(title="Sort By", options=cols, value=init_val)
#Column to use as a target for plotting
target = Select(title='Target', options=cols, value=init_val)
#Colun to use to set x axis
x_axis = Select(title='X axis', options=cols, value=default_x)
#Column to use to set y axis
y_axis = Select(title='Y axis', options=cols, value=default_y)
#Column to use for division
division_cat = Select(title='Subdivision Category', options=cols, value=default_div[0])
#Value to set division category equal to
division_val = TextInput(title='Subdivision Value (int)', value='1')
# #Linear or log based color mapper
# color_mapper = RadioGroup(labels=['Linear Scale', 'Log Scale'], active=0)
# color_mapper.on_change('active', lambda attr, old, new: update())
mini_val = TextInput(title='Minimum target value (float)', value='-inf')
maxi_val = TextInput(title='Maximum target value (float)', value='inf')
difference = Select(title='Difference From', options=cols, value=init_diff)

#Set how controls impact changes
controls = [sorted_by, target, x_axis, y_axis, division_cat, division_val, mini_val, maxi_val, difference]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

# controls.append(color_mapper)
### Build the final displace
inputs = widgetbox(*controls)
l = layout([
    [app.p, inputs]
])

#Do initial update of values
update()

#Set up document
curdoc().add_root(l)
curdoc().title = 'General Heatmap'
