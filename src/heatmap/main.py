'''This has the main execution of the heatmap app'''
import logging
import heatmap
import data_processing
from bokeh.plotting import curdoc
from bokeh.layouts import widgetbox, layout
from bokeh.models.widgets import Select, RadioGroup
logging.basicConfig()

#Build the source data
data = heatmap.HeatGrid()
app = heatmap.HeatMap(heatgrid=data)
app.build()

#Instantiate a dataframe to reference columns
df = data_processing.main()
#Identify only columns with integers or floats
cols = list(df.select_dtypes([int, float]).columns)


def update():
    '''This function updates the figure based on user inputs to the controls'''
    #Update the heatmap grid data
    app.heatgrid = heatmap.HeatGrid(x_axis=x_axis.value, y_axis=y_axis.value,
                                    sortby_x=sorted_by_x.value, sortby_y=sorted_by_y.value,
                                    target=target.value, normalization=normalization.active)
    #Use the new heatgrid parameter to update the heatmap
    app.update()


###Build the controls

#Column to sort the data by
sorted_by_x = Select(title="Sort X By", options=cols, value='CurRev')
sorted_by_y = Select(title="Sort Y By", options=cols, value='CurPrice')
#Column to use as a target for plotting
target = Select(title='Target', options=cols, value='CurPrice')
#Colun to use to set x axis
x_axis = Select(title='X axis', options=cols, value='ProductId')
#Column to use to set y axis
y_axis = Select(title='Y axis', options=cols, value='AreaId')
#Column to use for division
normalization = RadioGroup(labels=['No Normalization', 'Difference from column average',
                                   'Difference from row average'], active=1)
normalization.on_change('active', lambda attr, old, new: update())

#Set how controls impact changes
controls = [sorted_by_x, sorted_by_y, target, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

#Add in the Radiogroup
controls.append(normalization)

### Build the final display
inputs = widgetbox(*controls)
l = layout([
    [app.p, inputs]
])

#Do initial update of values
update()

#Set up Bokeh documents for backend html interactions
curdoc().add_root(l)
curdoc().title = 'General Heatmap'
