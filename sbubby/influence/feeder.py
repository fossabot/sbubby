#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K

"""
summary

description

:REQUIRES:

:TODO:

:AUTHOR: Matthew McAteer
:ORGANIZATION: UnifyID
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
import abc
import six

# =============================================================================
# METHODS
# =============================================================================

@six.add_metaclass(abc.ABCMeta)
class InfluenceFeeder:
    @abc.abstractmethod
    def reset(self):
        """ reset dataset
        """
        raise RuntimeError('must be implemented')

    @abc.abstractmethod
    def train_batch(self, batch_size):
        """ training data feeder by batch sampling

        Parameters
        ----------
        batch_size : batch size

        Returns
        -------
        xs : feed input values
        ys : feed label values

        """
        raise RuntimeError('must be implemented')

    @abc.abstractmethod
    def train_one(self, index):
        """ single training data feeder

        Parameters
        ----------
        index : training sample index

        Returns
        -------
        x : feed one input value
        y : feed one label value

        """
        raise RuntimeError('must be implemented')

    @abc.abstractmethod
    def test_indices(self, indices):
        """ test data feeder

        Parameters
        ----------
        indices : testing sample index

        Returns
        -------
        x : feed input values
        y : feed label values

        """
        raise RuntimeError('must be implemented')

class MNISTFeeder(InfluenceFeeder):
    def __init__(self):
        (train_data, train_label), (test_data, test_label) = mnist.load_data() 
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28
        self.train_origin_data = train_data
        self.train_origin_label = train_label
        # load test data
        self.test_origin_data = test_data
        self.test_origin_label = test_label
        
        if K.image_data_format() == 'channels_first':
            self.train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
        else:
            self.train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
        
        self.train_label = keras.utils.to_categorical(train_label, num_classes)
        self.test_label = keras.utils.to_categorical(test_label, num_classes)
        
        self.train_batch_offset = 0

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0

class CIFAR10Feeder(InfluenceFeeder):
    def __init__(self):
        (train_data, train_label), (test_data, test_label) = cifar10.load_data() 
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 32, 32
        self.train_origin_data = train_data
        self.train_origin_label = train_label
        # load test data
        self.test_origin_data = test_data
        self.test_origin_label = test_label
        
        if K.image_data_format() == 'channels_first':
            self.train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
        else:
            self.train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
        
        self.train_label = keras.utils.to_categorical(train_label, num_classes)
        self.test_label = keras.utils.to_categorical(test_label, num_classes)
        
        self.train_batch_offset = 0

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0

class CIFAR100Feeder(InfluenceFeeder):
    def __init__(self):
        (train_data, train_label), (test_data, test_label) = cifar100.load_data() 
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 32, 32
        self.train_origin_data = train_data
        self.train_origin_label = train_label
        # load test data
        self.test_origin_data = test_data
        self.test_origin_label = test_label
        
        if K.image_data_format() == 'channels_first':
            self.train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
        else:
            self.train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
        
        self.train_label = keras.utils.to_categorical(train_label, num_classes)
        self.test_label = keras.utils.to_categorical(test_label, num_classes)
        
        self.train_batch_offset = 0

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0


class FashionMNISTFeeder(InfluenceFeeder):
    def __init__(self):
        (train_data, train_label), (test_data, test_label) = fashion_mnist.load_data() 
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28
        self.train_origin_data = train_data
        self.train_origin_label = train_label
        # load test data
        self.test_origin_data = test_data
        self.test_origin_label = test_label
        
        if K.image_data_format() == 'channels_first':
            self.train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
        else:
            self.train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
            self.test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
        
        self.train_label = keras.utils.to_categorical(train_label, num_classes)
        self.test_label = keras.utils.to_categorical(test_label, num_classes)
        
        self.train_batch_offset = 0

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0
        
class CustomFeeder(InfluenceFeeder):
    def __init__(self, input_train_data, input_train_label, input_test_data, input_test_label, num_classes):
        # input image dimensions
        self.train_origin_data = input_train_data
        self.train_origin_label = input_train_label
        # load test data
        self.test_origin_data = input_test_data
        self.test_origin_label = input_test_label
        
        self.train_data = input_train_data
        self.test_data = input_test_data
        
        self.train_label = input_train_label
        self.test_label = input_test_label
        
        self.train_batch_offset = 0

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0
