#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Operations for influence candidates

description

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
import tensorflow as tf
import numpy as np

# =============================================================================
# METHODS
# =============================================================================

_unusable_op_names = (
    'Shape',
    'Reshape',
    'Slice',
    'Pack',
    'Cast',
    'ConcatV2',
    'Const',
    'Identity',
    'ZerosLike',
    'Assign',
    'VariableV2')


def _unusable_ops(op):
    if len(op.outputs) == 0 \
            or 'save' in op.name \
            or 'gradients/' in op.name \
            or '/Initializer' in op.name \
            or op.op_def is None \
            or op.op_def.name in _unusable_op_names:
        return True
    else:
        return False


def candidate_featuremap_op_names(sess, graph, feed_options):
    operations = []
    out_ranks = []
    out_shapes = []

    for op in graph.get_operations():
        if _unusable_ops(op):
            continue

        out_ranks.append(tf.rank(op.outputs[0]))
        out_shapes.append(tf.shape(op.outputs[0]))
        operations.append(op)

    out_ranks_val, out_shapes_val = sess.run([out_ranks, out_shapes], feed_dict=feed_options)

    ret = []
    for out_rank, out_shape, op in zip(out_ranks_val, out_shapes_val, operations):
        if out_rank != 4 or (out_shape[1] == 1 and out_shape[2] == 1) or out_shape[0] != 1:
            continue

        ret.append(op.name)
    return ret


def candidate_predict_op_names(sess, num_classes, graph, feed_options):
    operations = []
    out_ranks = []
    out_shapes = []

    for op in graph.get_operations():
        if _unusable_ops(op):
            continue

        out_ranks.append(tf.rank(op.outputs[0]))
        out_shapes.append(tf.shape(op.outputs[0]))
        operations.append(op)

    out_ranks_val, out_shapes_val = sess.run([out_ranks, out_shapes], feed_dict=feed_options)

    ret = []
    for out_rank, out_shape, op in zip(out_ranks_val, out_shapes_val, operations):
        if out_rank == 1:
            continue
        if np.prod(out_shape) != num_classes:
            continue

        ret.append(op.name)
    return ret

