'''This module contains the class for the heatmap object'''
from math import pi
import data_processing
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    ColorBar,
    BasicTicker
)
from bokeh.plotting import figure
import numpy as np

class HeatMap(object):
    '''This class contains information used in the heatmap'''

    def __init__(self, heatgrid=None, pal='Viridis256', title='Interactive Heatmap'):
        '''This initializes the class'''
        self.heatgrid = heatgrid
        self.pal = pal
        self.title = title
        self.p = generate_figure

    @property
    def generate_figure(self):
        '''This instantiates a figure object'''
        tools = "pan,wheel_zoom,reset,hover,save"
        p = figure(
            title=self.title, tools=tools,
            x_axis_location='above',
            toolbar_location='below'
        )
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = '12pt'
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 3
        p.xaxis.axis_label = self.heatgrid.x_axis
        p.yaxis.axis_label = self.heatgrid.y_axis

        self.build_mapper()

        p.rect(x='x', y='y', width=1, height=1,
                    source=self.source,
                    fill_color={'field': 'val', 'transform': self.mapper},
                    line_color=None)
        return p

    def hovertool(self):
        '''This method builds or updates the hovertool'''
        self.hover = self.p.select_one(HoverTool)
        self.hover.point_policy = "follow_mouse"
        self.hover.tooltips = [
            (self.heatgrid.x_axis, "@x_name"),
            (self.heatgrid.y_axis, "@y_name"),
            (self.heatgrid.target, "@val")
        ]

    def build_mapper(self):
        '''This method builds the mapper and ColorBar'''
        self.mapper = LinearColorMapper(palette=self.pal, low=min(self.source.data['val']),
                                        high=max(self.source.data['val']))
        self.color_bar = ColorBar(color_mapper=self.mapper, ticker=BasicTicker(),
                                  label_standoff=12, border_line_color=None, location=(0, 0))

    def build(self):
        '''This method runs all the steps necessary to build the plot'''
        data = self.heatgrid.return_data()
        self.source = ColumnDataSource(data)
        self.generate_figure()
        self.hovertool()
        self.p.add_layout(self.color_bar, 'right')

    def update(self):
        '''This method updates the plot with new features'''
        data = self.heatgrid.return_data()
        self.source.data = data
        self.mapper.high = max(data['val'])
        self.mapper.low = min(data['val'])
        self.color_bar.color_mapper.high = max(data['val'])
        self.color_bar.color_mapper.low = min(data['val'])
        self.hovertool()

class HeatGrid(object):
    '''This class is used to populat the grid of the heatmap'''

    def __init__(self, df=None, target='CurPrice', sortby_x='CurRev', sortby_y='CurPrice',
                 normalization=1, x_axis='ProductId', y_axis='AreaId',
                 selection_criteria={'SalesTypeId':1}, x_display='ProductDescription',
                 y_display='AreaId'):
        '''Initialize the class '''
        self.df = df
        self.target = target
        self.sortby_x = sortby_x
        self.sortby_y = sortby_y
        self.normalization = normalization
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.selection_criteria = selection_criteria
        self.x_display = x_display
        self.y_display = y_display

    def generate_grid(self):
        '''This method generates the x and y locations and the target for the
        heatmap grid, subject to minima, maxima, and normalization criteria
        Returns:
        x_locs -- x location for each point
        y_locs -- y location for each point
        target -- target value for each point'''

        x_locs = [self.interpret_xs[x] for x in self.df[self.x_axis].values]
        y_locs = [self.interpret_ys[y] for y in self.df[self.y_axis].values]

        #Grab the target without any normalization
        target = self.df[self.target].values
        if self.normalization == 1: #If normalizing across columns axis
            #Find the average target for each x value
            x_avgs = dict()
            for true_x, x_loc in self.interpret_xs.iteritems():
                x_avg = self.df[self.df[self.x_axis] == true_x][self.target].mean()
                x_avgs[x_loc] = x_avg
            #Subtract the average value from each point
            target = target - [x_avgs[x_loc] for x_loc in x_locs]
        elif self.normalization == 2: #If normalizing across rows axis
            #Find the average target for each y value
            y_avgs = dict()
            for true_y, y_loc in self.interpret_ys.iteritems():
                y_avg = self.df[self.df[self.y_axis] == true_y][self.target].mean()
                y_avgs[y_loc] = y_avg
            #Subtract the average value from each point
            target = target - [y_avgs[y_loc] for y_loc in y_locs]

        x_names = self.df[self.x_display]
        y_names = self.df[self.y_display]

        return x_locs, y_locs, x_names, y_names, target

    def grab_data(self, datafile='PriceBook.csv'):
        '''This method grabs new data and puts it in the dataframe'''
        self.df = data_processing.main(datafile)

    def build_axes_interpreters(self):
        '''This method builds the interpreters for the x and y axes'''
        raw_xs = self.df[self.x_axis].unique()
        raw_ys = self.df[self.y_axis].unique()

        total_x_targets = [self.df[self.df[self.x_axis] == raw_x][self.sortby_x].sum() for raw_x in raw_xs]
        total_y_targets = [self.df[self.df[self.y_axis] == raw_y][self.sortby_y].sum() for raw_y in raw_ys]

        sorted_x_inds = np.argsort(total_x_targets)
        sorted_y_inds = np.argsort(total_y_targets)

        self.interpret_xs = dict(zip(raw_xs[sorted_x_inds[::-1]], range(0, len(raw_xs))))
        self.interpret_ys = dict(zip(raw_ys[sorted_y_inds], range(0, len(raw_ys))))

    def generate_source_data(self):
        '''This method generates the source to be used in the plotting figure'''
        x_locs, y_locs, x_names, y_names, target = self.generate_grid()

        return dict(
            x=x_locs,
            y=y_locs,
            x_name=x_names,
            y_name=y_names,
            val=target,
        )

    def filter_data(self):
        '''This method cleans the dataframe using the provided selection criteria'''
        queries = []
        for key, val in self.selection_criteria.iteritems():
            queries.append('{} == {}'.format(key, val))
        query = ' and '.join(queries)
        self.df = self.df.query(query)

    def return_data(self):
        '''This method updates the data'''
        self.grab_data()
        self.filter_data()
        self.build_axes_interpreters()
        data = self.generate_source_data()
        return data
