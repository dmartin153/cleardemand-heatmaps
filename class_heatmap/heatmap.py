'''This module contains the class for the heatmap object'''
import data_processing
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    ColorBar,
    BasicTicker
)
import pdb
import confidential
from math import pi
from bokeh.plotting import figure


class Heatmap(object):
    '''This class contains information used in the heatmap'''

    def __init__(self, df=None, target='CurPrice', sortby='CurRev', normalization=0,
                x_axis='ProductId', y_axis='AreaId', min_tar_perc=0,
                max_tar_perc=1, selection_criteria= ('SalesTypeId',1), p=None):
        '''This initializes the class'''
        self.df = df
        self.target = target
        self.sortby = sortby
        self.normalization = normalization
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.min_tar_perc = min_tar_perc
        self.max_tar_perc = max_tar_perc
        self.selection_criteria = selection_criteria

    def generate_figure(self):
        '''This instantiates a figure object'''
        TOOLS = "pan,wheel_zoom,reset,hover,save"
        self.p = figure(
            title='Interactive Heatmap', tools=TOOLS,
            x_axis_location='above',
            toolbar_location='below'
        )
        self.p.grid.grid_line_color = None
        self.p.axis.axis_line_color = None
        self.p.axis.major_tick_line_color = None
        self.p.axis.major_label_text_font_size = '12pt'
        self.p.axis.major_label_standoff = 0
        self.p.xaxis.major_label_orientation = pi / 3

        self.build_mapper()

        self.p.rect(x='x', y='y', width=1, height=1,
                    source=self.source,
                    fill_color={'field': 'val', 'transform': self.mapper},
                    line_color=None)

    def grab_data(self):
        '''This method grabs new data and puts it in the dataframe'''
        self.df = data_processing.main()

    def hovertool(self):
        '''This method builds or updates the hovertool'''
        self.hover = self.p.select_one(HoverTool)
        self.hover.point_policy = "follow_mouse"
        self.hover.tooltips = [
            (self.x_axis, "@x_name"),
            (self.y_axis, "@y_name"),
            (self.target, "@val")
        ]

    def selection_data(self):
        '''This method cleans the dataframe using the provided selection criteria'''
        self.df = self.df[self.df[self.selection_criteria[0]] == self.selection_criteria[1]]
        self.df.sort_values(by=self.sortby, inplace=True)

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

    def build_axes_interpreters(self):
        '''This method builds the interpreters for the x and y axes'''
        raw_xs = self.df[self.x_axis].unique()
        raw_ys = self.df[self.y_axis].unique()

        self.interpret_xs = dict(zip(raw_xs,range(0,len(raw_xs))))
        self.interpret_ys = dict(zip(raw_ys,range(0,len(raw_ys))))

    def generate_grid(self):
        '''This method generates the x and y locations and the target for the
        heatmap grid, subject to minima, maxima, and normalization criteria
        Returns:
        x_locs -- x location for each point
        y_locs -- y location for each point
        target -- target value for each point'''

        df = self.df.copy()

        target = df[self.target].values

        x_locs = [self.interpret_xs[x] for x in df[self.x_axis].values]
        y_locs = [self.interpret_ys[y] for y in df[self.y_axis].values]

        x_names = df[self.x_axis]
        y_names = df[self.y_axis]

        return x_locs, y_locs, x_names, y_names, target

    def build_mapper(self):
        '''This method builds the mapper and ColorBar'''
        pal = 'Plasma256'
        self.mapper = LinearColorMapper(palette=pal, low=min(self.source.data['val']), high=max(self.source.data['val']))
        self.color_bar = ColorBar(color_mapper=self.mapper, ticker=BasicTicker(),
                                label_standoff=12, border_line_color=None, location=(0,0))

    def build(self):
        '''This method runs all the steps necessary to build the plot'''
        self.grab_data()
        self.selection_data()
        self.build_axes_interpreters()
        data = self.generate_source_data()
        self.source = ColumnDataSource(data)
        self.generate_figure()
        self.hovertool()

    def update(self):
        '''This method updates the plot with new features'''
        self.grab_data()
        self.selection_data()
        self.build_axes_interpreters()
        data = self.generate_source_data()
        self.source.data = data
        self.mapper.high = max(data['val'])
        self.mapper.low = min(data['val'])
        self.color_bar.color_mapper.high = max(data['val'])
        self.color_bar.color_mapper.low = min(data['val'])
        self.hovertool()
