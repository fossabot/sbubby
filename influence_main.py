# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
"""
Influence Function Class
Re-implementation of Koh et al., 2018.
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
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import os
import time
import hashlib
import json
from functools import wraps

import sbubby
from sbubby.influence.feeder import (MNISTFeeder, InfluenceFeeder,
                                     CIFAR10Feeder,
                                     FashionMNISTFeeder, CustomFeeder)
from sbubby.influence.influence import Influence
from sbubby.influence.plotter import (plot_unranked_influence,
                                      plot_ranked_influence)


# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================
def main(model_filename='mnist_cnn.h5', dataset='mnist',
         train_data_path='qmnist_x_train.npy',
         train_labels_path='qmnist_y_train.npy',
         test_data_path='qmnist_x_test.npy',
         test_labels_path='qmnist_y_test.npy',
         all_test='all', query_indices=99,
         testset_batch_size=100, train_batch_size=100,
         train_iterations=600, scale=1e5, damping=0.01, num_repeats=1,
         recursion_batch_size=100, recursion_depth=10000):
    """
    Main function for calculating influence scores for a
        dataset and the model that was trained on it.
    Args:
        model_name: path to .h5 file with model weights
        dataset: name of the dataset being used
        train_data: path to tensors for the training data
        train_labels: path to tensors for the training labels
        test_data: path to tensors for the test data
        test_labels: path to tensors for the test labels
        test_indices: the indexes of the training data for the influence
            function to be run on (if set to 'all', it will run influence
            calculations across the entire test dataset)
        testset_batch_size: batch size for the test data
        train_batch_size: batch size for the influence detector (should
            be equal to the batch sized used in the original model training
        train_iterations: The number of iterations (should be equal to the
            train data length divided by the batch size)')
        scale: Lower `scale` makes scores more exaggerated, higher does
            opposite (Too high can get nan norms, which results in random
            indexes and non-plottable scores)
        damping: Damping factor for models with non-convexities in their
            loss landscape
        num_repeats: Number of repeats of the influence calculation (more
            repeats will result in a closer approximation of the true
            influence scores, but will take longer)
        recursion_batch_size: Influence recursion batch size
        recursion_depth: Influence recursion depth
    """
    model = load_model(model_filename)
    if dataset == 'mnist':
        feeder = MNISTFeeder()
    if dataset == 'cifar10':
        feeder = CIFAR10Feeder()
    if dataset == 'cifar100':
        feeder = CIFAR100Feeder()
    if dataset == 'fashion_mnist':
        feeder = FashionMNISTFeeder()
    else:
        train_data = np.load(train_data_path)
        train_labels = np.load(train_labels_path)
        test_data = np.load(test_data_path)
        test_labels = np.load(test_labels_path)
        num_classes = train_labels.max()+1
        feeder = CustomFeeder(train_data, train_labels, test_data,
                              test_labels, num_classes)

    inspector = Influence(
        feeder=feeder,
        model=model)
    if all_test == 'all':
        test_indices = np.arange(0, len(feeder.test_origin_data))
    else:
        test_indices = query_indices
    testset_batch_size = testset_batch_size

    train_batch_size = train_batch_size
    train_iterations = train_iterations

    sess = tf.InteractiveSession()

    # Parameters for the influence function approximator itself:
    scale = scale
    damping = damping
    num_repeats = num_repeats
    recursion_batch_size = recursion_batch_size
    recursion_depth = recursion_depth
    approx_params = {
        'scale': scale,
        'damping': damping,
        'num_repeats': num_repeats,
        'recursion_batch_size': recursion_batch_size,
        'recursion_depth': recursion_depth
    }

    tf.global_variables_initializer().run()

    scores = inspector.upweighting_influence_batch(
        sess,
        test_indices,
        testset_batch_size,
        approx_params,
        train_batch_size,
        train_iterations)

    finishing_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    sorted_indices = np.argsort(scores)
    harmful = sorted_indices[:10]
    helpful = sorted_indices[-10:][::-1]

    print('\nHarmful:')
    for idx in harmful:
        print('[{}] {}'.format(idx, scores[idx]))

    print('\nHelpful:')
    for idx in helpful:
        print('[{}] {}'.format(idx, scores[idx]))

    pos_signal = scores[sorted_indices].copy()
    neg_signal = scores[sorted_indices].copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan

    # plotting
    plt.style.use('seaborn')
    plt.plot(pos_signal, color='g')
    plt.plot(neg_signal, color='r')
    plt.xlabel("Data ranking (lowest to highest helpfulness)")
    plt.ylabel("Influence Score")
    plt.title("Influence Scores by ranking\nnum_test_indices:{}, testset_batch_size:{}, train_batch_size:{}, train_iterations:{}\nscale:{}, damping:{}, num_repeats:{}, recursion_batch_size:{}, recursion_depth:{}".format(
        len(test_indices),
        testset_batch_size,
        train_batch_size,
        train_iterations,
        approx_params['scale'],
        approx_params['damping'],
        approx_params['num_repeats'],
        approx_params['recursion_batch_size'],
        approx_params['recursion_depth']))
    plt.savefig('pos_neg.png', dpi=1000)

    influence_results = pd.DataFrame(
            data=np.transpose(
                    np.stack(
                            [scores[sorted_indices],
                             [i.astype(int) for i in sorted_indices],
                             [i.astype(int) for i in feeder.train_origin_label[sorted_indices]],
                             np.arange(len(scores), 0, -1),
                             np.arange(1, len(scores)+1, 1)
                             ]
                            )
                    ),
            columns=["score",
                     "index",
                     "label",
                     "pos_rank",
                     "neg_rank"])

    influence_results.to_csv(
            '{}_influence_results_{}.csv'.format(model_filename,
                                                 finishing_time))
    plot_unranked_influence(
            '{}_influence_results_{}.csv'.format(model_filename,
                                                 finishing_time))
    plot_ranked_influence(
            '{}_influence_results_{}.csv'.format(model_filename,
                                                 finishing_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    # Model and dataset details
    parser.add_argument('--model_name', '-n', default='mnist_cnn.h5', type=str,
                        help='path to .h5 file with model weights')
    parser.add_argument('--dataset', '-t', default='mnist',
                        help='name of the dataset being used \n( mnist | cifar10 | cifar100 | fashion_mnist | qmnist | padova | gaitnet )')
    parser.add_argument('--train_data', '-x', default='X_padova_train.npy',
                        help='path to tensors for the training data')
    parser.add_argument('--train_labels', '-y', default='y_padova_train.npy',
                        help='path to tensors for the training labels')
    parser.add_argument('--test_data', '-v', default='X_padova_test.npy',
                        help='path to tensors for the test data')
    parser.add_argument('--test_labels', '-w', default='y_padova_test.npy',
                        help='path to tensors for the test labels')
    # Influence Training parameters
    parser.add_argument('--all_test', '-a', default='all',
                        help='whether influence calculations should be run on all the test data (if set to `all`, it will run influence calculations across the entire test dataset)')
    parser.add_argument('--query_indices', '-i', default=99,
                        help='the indexes of the test data for the influence function to be run on (if `all_test` is set to `all`, this will be ignored)')
    parser.add_argument('--testset_batch_size', '-b', default=100,
                        help='batch size for the test data')
    parser.add_argument('--train_batch_size', '-k', default=100,
                        help='batch size for the influence detector (should be equal to the batch sized used in the original model training')
    parser.add_argument('--train_iterations', '-l', default=600,
                        help='The number of iterations (should be equal to the train data length divided by the batch size)')
    # Influence detector parameters
    parser.add_argument('--scale', '-s', default=1e5,
                        help='Lower `scale` makes scores more exaggerated, higher does opposite (Too high can get nan norms, which results in random indexes and non-plottable scores)')
    parser.add_argument('--damping', '-d', default=0.01,
                        help='Damping factor for models with non-convexities in their loss landscape')
    parser.add_argument('--num_repeats', '-m', default=1,
                        help='Number of repeats of the influence calculation (more repeats will result in a closer approximation of the true influence scores, but will take longer)')
    parser.add_argument('--recursion_batch_size', '-r', default=100,
                        help='Influence recursion batch size')
    parser.add_argument('--recursion_depth', '-p', default=10000,
                        help='Influence recursion depth')

    args = parser.parse_args()

    main(model_filename=args.model_name, dataset=args.dataset,
         train_data_path=args.train_data,
         train_labels_path=args.train_labels,
         test_data_path=args.test_data,
         test_labels_path=args.test_labels,
         all_test=args.all_test, query_indices=args.query_indices,
         testset_batch_size=args.testset_batch_size,
         train_batch_size=args.train_batch_size,
         train_iterations=args.train_iterations,
         scale=args.scale, damping=args.damping,
         num_repeats=args.num_repeats,
         recursion_batch_size=args.recursion_batch_size,
         recursion_depth=args.recursion_depth)
