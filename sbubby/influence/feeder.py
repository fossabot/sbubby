#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

"""
Data feeders for influence detection (including random mislabels).
Contains built-in classes for mnist, cifar10, cifar100, and fashion_mnist

:REQUIRES: keras, tensorflow, numpy, abc, six
:AUTHOR: Matthew McAteer
:CONTACT: matthewmcateer0@gmail.com
:SINCE: Sat Jun 8 14:43:35 2019
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
import keras
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

# =============================================================================
# METHODS
# =============================================================================

# cifar-10 classes
_mnist_classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                  'eight', 'nine')
_cifar10_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                    'frog','horse','ship','truck')
#_cifar100_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
#                    'frog','horse','ship','truck')
_fmnist_classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

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
    def __init__(self, mislabel=False, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        from keras.datasets import mnist
        (train_data, train_label), (test_data, test_label) = mnist.load_data() 
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28
        self.train_origin_data = train_data
        if mislabel==True:
            train_label = self.make_mislabel(train_label, 
                                             target_class_idx=7,
                                             true_class_idx=5,
                                             mislabel_rate=0.01)
            self.train_origin_label = train_label
        else:
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

    def make_mislabel(self, label, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        """Take a random `mislabel_rate` of the `true_class_idx` and change it to `target_class_idx`"""
        correct_indices = np.where(label == target_class_idx)[0]       
        self.correct_indices = correct_indices[:]
        labeled_true = np.where(label == true_class_idx)[0]
        np.random.shuffle(labeled_true)
        mislabel_indices = labeled_true[:int(labeled_true.shape[0] * mislabel_rate)]
        label[mislabel_indices] = float(target_class_idx)
        self.mislabel_indices = mislabel_indices

        print('target class: {}'.format(_mnist_classes[target_class_idx]))
        print(self.mislabel_indices)
        return label

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
    def __init__(self, mislabel=False, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        from keras.datasets import cifar10
        (train_data, train_label), (test_data, test_label) = cifar10.load_data()
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 32, 32
        self.train_origin_data = train_data
        if mislabel == True:
            train_label = self.make_mislabel(train_label, 
                                             target_class_idx=7,
                                             true_class_idx=5,
                                             mislabel_rate=0.01)
            self.train_origin_label = train_label
        else:
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

    def make_mislabel(self, label, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        """Take a random `mislabel_rate` of the `true_class_idx` and change it to `target_class_idx`"""
        correct_indices = np.where(label == target_class_idx)[0]       
        self.correct_indices = correct_indices[:]
        labeled_true = np.where(label == true_class_idx)[0]
        np.random.shuffle(labeled_true)
        mislabel_indices = labeled_true[:int(labeled_true.shape[0] * mislabel_rate)]
        label[mislabel_indices] = float(target_class_idx)
        self.mislabel_indices = mislabel_indices

        print('target class: {}'.format(_cifar10_classes[target_class_idx]))
        print(self.mislabel_indices)
        return label

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

#class CIFAR100Feeder(InfluenceFeeder):
#    def __init__(self, mislabel=False, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
#        from keras.datasets import cifar100
#        (train_data, train_label), (test_data, test_label) = cifar100.load_data() 
#        num_classes = 10
#        # input image dimensions
#        img_rows, img_cols = 32, 32
#        self.train_origin_data = train_data
#        if mislabel == True:
#            train_label = self.make_mislabel(train_label)
#            self.train_origin_label = train_label
#        else:
#            self.train_origin_label = train_label
#        # load test data
#        self.test_origin_data = test_data
#        self.test_origin_label = test_label
#        
#        if K.image_data_format() == 'channels_first':
#            self.train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
#            self.test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols).astype('float32')/255
#        else:
#            self.train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
#            self.test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1).astype('float32')/255
#        
#        self.train_label = keras.utils.to_categorical(train_label, num_classes)
#        self.test_label = keras.utils.to_categorical(test_label, num_classes)
#        
#        self.train_batch_offset = 0
#
#    def make_mislabel(self, label, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
#        """Take a random `mislabel_rate` of the `true_class_idx` and change it to `target_class_idx`"""
#        correct_indices = np.where(label == target_class_idx)[0]       
#        self.correct_indices = correct_indices[:]
#        labeled_true = np.where(label == true_class_idx)[0]
#        np.random.shuffle(labeled_true)
#        mislabel_indices = labeled_true[:int(labeled_true.shape[0] * mislabel_rate)]
#        label[mislabel_indices] = float(target_class_idx)
#        self.mislabel_indices = mislabel_indices
#
#        print('target class: {}'.format(_cifar100_classes[target_class_idx]))
#        print(self.mislabel_indices)
#        return label
#
#    def test_indices(self, indices):
#        return self.test_data[indices], self.test_label[indices]
#
#    def train_batch(self, batch_size):
#        # calculate offset
#        start = self.train_batch_offset
#        end = start + batch_size
#        self.train_batch_offset += batch_size
#
#        return self.train_data[start:end, ...], self.train_label[start:end, ...]
#
#    def train_one(self, idx):
#        return self.train_data[idx, ...], self.train_label[idx, ...]
#
#    def reset(self):
#        self.train_batch_offset = 0


class FashionMNISTFeeder(InfluenceFeeder):
    def __init__(self, mislabel=False, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        from keras.datasets import fashion_mnist
        (train_data, train_label), (test_data, test_label) = fashion_mnist.load_data() 
        num_classes = 10
        # input image dimensions
        img_rows, img_cols = 28, 28
        self.train_origin_data = train_data
        if mislabel == True:
            train_label = self.make_mislabel(train_label,
                                             target_class_idx=7,
                                             true_class_idx=5,
                                             mislabel_rate=0.01)
            self.train_origin_label = train_label
        else:
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

    def make_mislabel(self, label, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        """Take a random `mislabel_rate` of the `true_class_idx` and change it to `target_class_idx`"""
        correct_indices = np.where(label == target_class_idx)[0]       
        self.correct_indices = correct_indices[:]
        labeled_true = np.where(label == true_class_idx)[0]
        np.random.shuffle(labeled_true)
        mislabel_indices = labeled_true[:int(labeled_true.shape[0] * mislabel_rate)]
        label[mislabel_indices] = float(target_class_idx)
        self.mislabel_indices = mislabel_indices

        print('target class: {}'.format(_fmnist_classes[target_class_idx]))
        print(self.mislabel_indices)
        return label

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
    def __init__(self, input_train_data, input_train_label, input_test_data,
                 input_test_label, num_classes, mislabel=False,
                 target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        self.train_origin_data = input_train_data
        if mislabel == True:
            input_train_label = self.make_mislabel(input_train_label, 
                                                   target_class_idx=7,
                                                   true_class_idx=5,
                                                   mislabel_rate=0.01)
            self.train_origin_label = input_train_label
        else:
            self.train_origin_label = input_train_label

        # load test data
        self.test_origin_data = input_test_data
        self.test_origin_label = input_test_label
        
        self.train_data = self.train_origin_data
        self.test_data = self.test_origin_data 
        self.train_label = self.train_origin_label
        self.test_label = self.test_origin_label

        self.train_batch_offset = 0

    def make_mislabel(self, label, target_class_idx=7, true_class_idx=5, mislabel_rate=0.01):
        """Take a random `mislabel_rate` of the `true_class_idx` and change it to `target_class_idx`"""
        correct_indices = np.where(label == target_class_idx)[0]       
        self.correct_indices = correct_indices[:]
        labeled_true = np.where(label == true_class_idx)[0]
        np.random.shuffle(labeled_true)
        mislabel_indices = labeled_true[:int(labeled_true.shape[0] * mislabel_rate)]
        label[mislabel_indices] = float(target_class_idx)
        self.mislabel_indices = mislabel_indices

        print('target class: {}'.format(_classes[target_class_idx]))
        print(self.mislabel_indices)
        return label

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
