#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
summary
description

:REQUIRES:

:TODO:

:AUTHOR: Matthew McAteer
:SINCE: Wed Jun 12 13:30:50 2019
:VERSION: 0.1
"""
# =============================================================================
# PROGRAM METADATA
# =============================================================================
__author__ = 'Matthew McAteer'
__contact__ = 'matthewmcateer0@gmail.com'
__license__ = ''
__date__ = 'Wed Jun 12 13:30:50 2019'
__version__ = '0.1'

# =============================================================================
# IMPORT STATEMENTS
# =============================================================================
import os

# =============================================================================
# METHODS
# =============================================================================

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 31, 31
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    elif dataset == 'mnist':
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    elif dataset == 'ginger3':
        x_train, y_train = get_data('train')
        x_test,  y_test  = get_data('test')
        x_train = x_train / 255
        x_test  = x_test / 255
        y_train = keras.utils.to_categorical(y_train, num_classes = 3)
        y_test  = keras.utils.to_categorical(y_test, num_classes = 3)
            
    elif dataset == 'herb7':
        x_train, y_train = get_data('train')
        x_test,  y_test  = get_data('test')
        x_train = x_train / 255
        x_test  = x_test / 255
        y_train = keras.utils.to_categorical(y_train, num_classes = 7)
        y_test  = keras.utils.to_categorical(y_test, num_classes = 7)

    elif dataset == 'padova':
        x_train = np.load('X_1_1k_train.py')
        x_test = np.load('X_1_1k_train.py')
        y_train = np.load('y_1_1k_test.py')
        y_test = np.load('y_1_1k_test.py')
    elif dataset == 'gaitnet':
        x_train = np.load('X_1_1k_train.py')
        x_test = np.load('X_1_1k_train.py')
        y_train = np.load('y_1_1k_test.py')
        y_test = np.load('y_1_1k_test.py')
