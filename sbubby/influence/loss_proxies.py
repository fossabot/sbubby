#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
"""
Replacements for loss functions so they are twice-differentiable
Adapted from the TF Repo
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

import collections
import numbers
import numpy as np
import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export

# =============================================================================
# METHODS
# =============================================================================

@tf.custom_gradient
def sparse_softmax_cross_entropy_with_logits_with_gradients(labels, logits):
    """ Copied from tf.nn.sparse_softmax_cross_entropy_with_logits_with_gradients """
    with tf.name_scope("SparseSoftmaxCrossEntropyWithLogits"):
        labels = tf.convert_to_tensor(labels)
        logits = tf.convert_to_tensor(logits)
        precise_logits = tf.cast(logits, tf.float32) if (tf.as_dtype(logits.dtype) == tf.float16) else logits

        # Store label shape for result later.
        labels_static_shape = labels.get_shape()
        labels_shape = tf.shape(labels)
        static_shapes_fully_defined = (
                labels_static_shape.is_fully_defined() and
                logits.get_shape()[:-1].is_fully_defined())
        if logits.get_shape().ndims is not None and logits.get_shape().ndims == 0:
            raise ValueError(
                "Logits cannot be scalars - received shape %s." % logits.get_shape())
        if logits.get_shape().ndims is not None and (
                labels_static_shape.ndims is not None and
                labels_static_shape.ndims != logits.get_shape().ndims - 1):
            raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                             "equal rank of logits minus 1 (received %s)." %
                             (labels_static_shape.ndims, logits.get_shape().ndims))
        if (static_shapes_fully_defined and
                labels_static_shape != logits.get_shape()[:-1]):
            raise ValueError("Shape mismatch: The shape of labels (received %s) "
                             "should equal the shape of logits except for the last "
                             "dimension (received %s)." % (labels_static_shape,
                                                           logits.get_shape()))
        # Check if no reshapes are required.
        if logits.get_shape().ndims == 2:
            cost, dcost_dlogits = _tf_sotfmax_with_grads(precise_logits, labels)
            if logits.dtype == tf.float16:
                cost = tf.cast(cost, tf.float16)
                dcost_dlogits = tf.cast(dcost_dlogits, tf.float16)

            def grad(dcost, d2cost_dlogits):
                return None, grad_of_sparse_softmax_cross_entropy_with_logits(logits, dcost_dlogits, dcost)

            return (cost, dcost_dlogits), grad

        # Perform a check of the dynamic shapes if the static shapes are not fully
        # defined.
        shape_checks = []
        if not static_shapes_fully_defined:
            shape_checks.append(
                tf.assert_equal(
                    tf.shape(labels),
                    tf.shape(logits)[:-1]))
        with tf.control_dependencies(shape_checks):
            # Reshape logits to 2 dim, labels to 1 dim.
            num_classes = tf.shape(logits)[tf.rank(logits) - 1]
            precise_logits = tf.reshape(precise_logits, [-1, num_classes])
            labels = tf.reshape(labels, [-1])
            # The second output tensor contains the gradients.  We use it in
            # _CrossEntropyGrad() in nn_grad but not here.
            cost, dcost_dlogits = _tf_sotfmax_with_grads(precise_logits, labels)
            cost = tf.reshape(cost, labels_shape)
            cost.set_shape(labels_static_shape)
            dcost_dlogits = tf.reshape(dcost_dlogits, logits.shape)
            dcost_dlogits.set_shape(logits.shape)
            if logits.dtype == tf.float16:
                cost = tf.cast(cost, tf.float16)
                dcost_dlogits = tf.cast(dcost_dlogits, tf.float16)

            def grad(dcost, d2cost_dlogits):
                return None, grad_of_sparse_softmax_cross_entropy_with_logits(logits, dcost_dlogits, dcost)

            return (cost, dcost_dlogits), grad


@tf.custom_gradient
def grad_of_sparse_softmax_cross_entropy_with_logits(logits, dcost_dlogits, dcost):
    dcost = tf.expand_dims(dcost, axis=-1)

    def grad(dy):
        p = tf.nn.softmax(logits)
        d2logits = p * dy - p * tf.reduce_sum(p * dy, axis=-1, keepdims=True)
        return d2logits * dcost, None, dcost_dlogits * dy
    return dcost_dlogits * dcost, grad


def sparse_softmax_cross_entropy_with_logits(labels, logits):
    return sparse_softmax_cross_entropy_with_logits_with_gradients(labels, logits)[0]
