#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

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
