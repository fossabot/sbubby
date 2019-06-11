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
from keras import backend as K
import os
import time
import hashlib
import json
from functools import wraps

import influence_debugger
from influence_debugger.influence.feeder import MNISTFeeder
from influence_debugger.influence.influence import Influence

# =============================================================================
# METHODS
# =============================================================================



# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================
def main(model_filename, data):
    """Description of main()"""
    model = load_model(model_filename)
    
    if data == 'mnist':
        feeder = MNISTFeeder()
    else:
        feeder = MNISTFeeder()

    inspector = Influence(
        feeder=feeder,
        model=model)
    test_indices = np.arange(0,len(feeder.test_origin_data))
    testset_batch_size = 100 #@param {type:"integer"}

    train_batch_size = 100 #@param {type:"integer"}
    train_iterations = 600 #@param {type:"integer"}

    sess = tf.InteractiveSession()

    #@markdown **Parameters for the influence function approximator itself:**
    scale = 1e5   #@param 
    #@markdown Lower **`scale`** makes scores more exaggerated, higher does opposite
    #@markdown (Too high can get nan norms, which results in random indexes and non-plottable scores)
    damping = 0.01 #@param 
    num_repeats = 1 #@param 
    recursion_batch_size = 100 #@param 
    recursion_depth = 10000 #@param 
    approx_params = {
        'scale': scale,   # lower makes scores more exaggerated, higher does opposite (but too high can get nan norms, which results in random indexes and non-plottable scores)
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

    #plotting
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
             np.arange(len(scores),0,-1),
             np.arange(1,len(scores)+1,1)
            ])),
    columns=["score", "index", "label", "pos_rank", "neg_rank"])
    
    influence_results.to_csv('{}_influence_results_{}.csv'.format(model_filename, finishing_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--model_name', '-m', default='examples/mnist_cnn.h5', type=str, help='path to .h5 file with model weights')
    parser.add_argument('--train_data', '-t', default='mnist', help='tensors for the training data')
    parser.add_argument('--test_data', '-r', help='tensors for the test dats')
    args = parser.parse_args()
    
    main(args.model_name, args.train_data)
