# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
"""
Bokeh plotting for influence function outputs
:REQUIRES:
:TODO:
:AUTHOR: Matthew McAteer
:CONTACT: matthewmcateer0@gmail.com
:SINCE: Sat Jun  8 14:43:35 2019
:VERSION: 0.1
"""
# =============================================================================
# PROGRAM METADATA
# =============================================================================
__author__ = 'Matthew McAteer'
__contact__ = 'matthewmcateer0@gmail.com'
__license__ = ''
__date__ = 'Sat Jun  8 14:43:35 2019'
__version__ = '0.1'

# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import argparse
import datetime
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K
import os
import time
import hashlib
import json
from functools import wraps
from bokeh.plotting import output_file, save
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import brewer
from bokeh.models import (
    ColumnDataSource,
    ColorBar,
    LinearColorMapper,
    LogColorMapper,
)
from bokeh.models.annotations import BoxAnnotation
from bokeh.models.tools import HoverTool
from bokeh.palettes import Viridis3, Viridis256, Category10, Plasma256
from bokeh.plotting import figure
from bokeh.transform import transform

# =============================================================================
# METHODS
# =============================================================================
def plot_unranked_influence(csv_filename):

    influence_results = pd.read_csv(csv_filename) 


    # Create a ColumnDataSource from df: source
    source = ColumnDataSource(influence_results)

    TOOLS = "box_select,lasso_select,help,pan,wheel_zoom,box_zoom,reset"
    TITLE = "Influence scores for training data (colored and ordered by helpfulness)"

    p = figure(tools=TOOLS, toolbar_location="above",
               plot_width=800, plot_height=400, title=TITLE)
    p.toolbar.logo = "normal"

    color_mapper = LinearColorMapper(palette=Plasma256, 
                                     low=np.min(influence_results['label'].values), 
                                     high=np.max(influence_results['label'].values))

    # region that always fills the top of the plot
    upper = BoxAnnotation(bottom=0, fill_alpha=0.025, fill_color='green')
    p.add_layout(upper)

    # region that always fills the bottom of the plot
    lower = BoxAnnotation(top=0, fill_alpha=0.025, fill_color='firebrick')
    p.add_layout(lower)

    color_bar = ColorBar(color_mapper=color_mapper, location=(0,0))
    p.add_layout(color_bar, 'right')

    # add a circle renderer with x and y coordinates, size, color, and alpha
    cr = p.circle('index', 'score', size=5,
                  fill_color={'field': 'label', 'transform': color_mapper}, 
                  hover_fill_color="orange",
                  line_color={'field': 'label', 'transform': color_mapper}, 
                  hover_line_color="orange",
                  fill_alpha=1.0, hover_fill_alpha=1.0, source=source)

    p.add_tools(HoverTool(tooltips=[("score", "@score"),
                                    ('index', '@index'),
                                    ("label", "@label"),
                                    ('pos_rank', '@pos_rank'),
                                    ("neg_rank", "@neg_rank")],
                          renderers=[cr],
                          mode='mouse'))

    p.outline_line_width = 3
    p.outline_line_alpha = 0.15
    p.outline_line_color = "navy"
    p.xaxis.axis_label = 'Index (unranked)'
    p.yaxis.axis_label = 'Influence Score'

    output_file("{}_unranked_influence.html".format(csv_filename))
    save(p)


def plot_ranked_influence(csv_filename):
    
    influence_results = pd.read_csv(csv_filename) 
    # Create a ColumnDataSource from df: source
    source = ColumnDataSource(influence_results)

    TOOLS = "box_select,lasso_select,help,pan,wheel_zoom,box_zoom,reset"
    TITLE = "Influence scores for training data (colored and ordered by helpfulness)"

    p = figure(tools=TOOLS, toolbar_location="above",
               plot_width=800, plot_height=400, title=TITLE)
    p.toolbar.logo = "normal"

    color_mapper = LinearColorMapper(palette=Viridis256, 
                                     low=np.min(influence_results['label'].values), 
                                     high=np.max(influence_results['label'].values))

    # region that always fills the top of the plot
    upper = BoxAnnotation(bottom=0, fill_alpha=0.025, fill_color='green')
    p.add_layout(upper)

    # region that always fills the bottom of the plot
    lower = BoxAnnotation(top=0, fill_alpha=0.025, fill_color='firebrick')
    p.add_layout(lower)

    color_bar = ColorBar(color_mapper=color_mapper, location=(0,0))
    p.add_layout(color_bar, 'right')

    # add a circle renderer with x and y coordinates, size, color, and alpha
    cr = p.circle('neg_rank', 'score', size=5,
                  fill_color={'field': 'label', 'transform': color_mapper}, 
                  hover_fill_color="orange",
                  line_color={'field': 'label', 'transform': color_mapper}, 
                  hover_line_color="orange",
                  fill_alpha=1.0, hover_fill_alpha=1.0, source=source)

    p.add_tools(HoverTool(tooltips=[("score", "@score"),
                                    ('index', '@index'),
                                    ("label", "@label"),
                                    ('pos_rank', '@pos_rank'),
                                    ("neg_rank", "@neg_rank")],
                          renderers=[cr],
                          mode='mouse'))

    p.outline_line_width = 3
    p.outline_line_alpha = 0.15
    p.outline_line_color = "navy"
    p.xaxis.axis_label = 'Ranked Index (1=lowest score)'
    p.yaxis.axis_label = 'Influence Score'

    output_file("{}_unranked_influence.html".format(csv_filename))
    save(p)


# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================
def main(influence_records):
    """Description of main()"""
    plot_unranked_influence(influence_records)
    plot_ranked_influence(influence_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--influence_records', '-i', type=str, help='path to .csv file with influence output')
    args = parser.parse_args()
    
    main(args.influence_records)
