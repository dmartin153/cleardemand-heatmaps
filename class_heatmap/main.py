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
data = heatmap.HeatGrid()
app = heatmap.HeatMap(heatgrid=data)
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
init_sort = confidential.sort_columns()[0]

def update():
    '''This method updates the figure based on user inputs to the controls'''
    #Build a new source
    app.heatgrid = heatmap.HeatGrid(x_axis=x_axis.value, y_axis=y_axis.value,
        sortby = sorted_by.value, target = target.value, normalization=normalization.active)
    app.update()


###Make the controls
#Column to sort the data by
sorted_by = Select(title="Sort By", options=cols, value=init_sort)
#Column to use as a target for plotting
target = Select(title='Target', options=cols, value=init_val)
#Colun to use to set x axis
x_axis = Select(title='X axis', options=cols, value=default_x)
#Column to use to set y axis
y_axis = Select(title='Y axis', options=cols, value=default_y)
#Column to use for division
normalization = RadioGroup(labels=['No Normalization', 'Difference from column average', 'Difference from row average'], active=0)
normalization.on_change('active', lambda attr, old, new: update())
#Set how controls impact changes
controls = [sorted_by, target, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

controls.append(normalization)
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
