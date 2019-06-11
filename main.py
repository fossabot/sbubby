#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
"""
Simple influence calculation runner

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
from .feeder import InfluenceFeeder  # noqa: ignore=F401
from ..log import logger

import numpy as np
import tensorflow as tf
import keras
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

_using_fully_tf = True

kerasLossDict = {
    'categorical_crossentropy': keras.losses.categorical_crossentropy,
    'sparse_categorical_crossentropy': keras.losses.sparse_categorical_crossentropy,
    'binary_crossentropy': keras.losses.binary_crossentropy,
    'kullback_leibler_divergence': keras.losses.kullback_leibler_divergence,
    'poisson': keras.losses.poisson,
    'cosine_proximity': keras.losses.cosine_proximity
}

# =============================================================================
# METHODS
# =============================================================================

def _timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug('* %s function took [%.3fs]' % (f.__name__, time2-time1))
        return ret
    return wrap


class Influence:
    """ Influence Class (for keras models)
    """
    def __init__(self, feeder, model):
        self.workspace = './influence-workspace'
        self.feeder = feeder
        self.x_placeholder = model.input
        self.y_placeholder = K.placeholder(shape=model.output.shape)
        self.test_feed_options = dict()
        self.train_feed_options = dict()
        
        if model.loss in kerasLossDict.keys():
            loss_op_train = kerasLossDict[model.loss](self.y_placeholder, model.output)
            loss_op_test = kerasLossDict[model.loss](self.y_placeholder, model.output)
        else:
            loss_op_train = model.loss(self.y_placeholder, model.output)
            loss_op_test = model.loss(self.y_placeholder, model.output)

        trainable_variables = model.trainable_weights

        self.loss_op_train = loss_op_train
        self.grad_op_train = K.gradients(loss_op_train, trainable_variables)
        self.grad_op_test = K.gradients(loss_op_test, trainable_variables)

        self.v_cur_estimated = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in trainable_variables]
        self.v_test_grad = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in trainable_variables]
        self.v_ihvp = tf.placeholder(tf.float64, shape=[None])
        self.v_param_damping = tf.placeholder(tf.float32)
        self.v_param_scale = tf.placeholder(tf.float32)
        self.v_param_total_trainset = tf.placeholder(tf.float64)

        self.inverse_hvp = None
        self.trainable_variables = trainable_variables

        with tf.name_scope('model_ihvp'):
            self.hessian_vector_op = _hessian_vector_product(loss_op_train, trainable_variables, self.v_cur_estimated)
            self.estimation_op = [
                a + (b * self.v_param_damping) - (c / self.v_param_scale)
                for a, b, c in zip(self.v_test_grad, self.v_cur_estimated, self.hessian_vector_op)
            ]

        with tf.name_scope('model_grad_diff'):
            flatten_inverse_hvp = tf.reshape(self.v_ihvp, shape=(-1, 1))
            flatten_grads = tf.concat([tf.reshape(a, (-1,)) for a in self.grad_op_train], 0)
            flatten_grads = tf.reshape(flatten_grads, shape=(1, -1,))
            flatten_grads = tf.cast(flatten_grads, tf.float64)
            flatten_grads /= self.v_param_total_trainset
            self.grad_diff_op = tf.matmul(flatten_grads, flatten_inverse_hvp)

        self.ihvp_config = {
            'scale': 1e4,
            'damping': 0.01,
            'num_repeats': 1,
            'recursion_batch_size': 10,
            'recursion_depth': 10000
        }

        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)


    @_timing
    def upweighting_influence(self, sess, test_indices, test_batch_size, approx_params,
                              train_indices, num_total_train_example, force_refresh=False):
        """ Calculate influence score of given training samples that affect on the test samples
         Negative value indicates bad effect on the test loss
        """
        self._prepare(sess, test_indices, test_batch_size, approx_params, force_refresh)

        self.feeder.reset()
        score = self._grad_diffs(sess, train_indices, num_total_train_example)
        logger.info('Multiplying by %s train examples' % score.size)
        return score


    @_timing
    def upweighting_influence_batch(self, sess, test_indices, test_batch_size, approx_params,
                                    train_batch_size, train_iterations, subsamples=-1, force_refresh=False):
        """ Iteratively calculate influence scores for training data sampled by batch sampler
        Negative value indicates bad effect on the test loss
        """
        self._prepare(sess, test_indices, test_batch_size, approx_params, force_refresh)

        self.feeder.reset()
        score = self._grad_diffs_all(sess, train_batch_size, train_iterations, subsamples)
        logger.info('Multiplying by %s train examples' % score.size)
        return score


    @_timing
    def _prepare(self, sess, test_indices, test_batch_size, approx_params, force_refresh):
        """ Calculate inverse hessian vector product, and save it in workspace
        """
        # update ihvp approx params
        if approx_params is not None:
            for param_key in approx_params.keys():
                if param_key not in self.ihvp_config:
                    raise RuntimeError('unknown ihvp config param is approx_params')
            self.ihvp_config.update(approx_params)

        inv_hvp_path = self._path(self._approx_filename(sess, test_indices))
        if not os.path.exists(inv_hvp_path) or force_refresh:
            self.feeder.reset()
            test_grad_loss = self._get_test_grad_loss(sess, test_indices, test_batch_size)
            logger.info('Norm of test gradient: %s' % np.linalg.norm(np.concatenate([a.reshape(-1) for a in test_grad_loss])))
            self.inverse_hvp = self._get_inverse_hvp_lissa(sess, test_grad_loss)
            np.savez(inv_hvp_path, inverse_hvp=self.inverse_hvp, encoding='bytes')
            logger.info('Saved inverse HVP to %s' % inv_hvp_path)
        else:
            self.inverse_hvp = np.load(inv_hvp_path, encoding='bytes')['inverse_hvp']
            logger.info('Loaded inverse HVP from %s' % inv_hvp_path)


    def _get_test_grad_loss(self, sess, test_indices, test_batch_size):
        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / test_batch_size))
            test_grad_loss = None
            for i in range(num_iter):
                start = i * test_batch_size
                end = int(min((i + 1) * test_batch_size, len(test_indices)))
                size = float(end - start)

                test_feed_dict = self._make_test_feed_dict(*self.feeder.test_indices(test_indices[start:end]))
                temp = sess.run(self.grad_op_test, feed_dict=test_feed_dict)
                temp = np.asarray(temp)

                temp *= size
                if test_grad_loss is None:
                    test_grad_loss = temp
                else:
                    test_grad_loss += temp

            test_grad_loss /= len(test_indices)
        else:
            raise RuntimeError('unsupported yet')
        return test_grad_loss

    
    def _approx_filename(self, sess, test_indices):
        sha = hashlib.sha1()

        # weights
        vs = sess.run(self.trainable_variables)
        for a in vs:
            sha.update(a.data)

        # test_indices
        np_test_indices = np.array(list(test_indices))
        sha.update(np_test_indices.data)

        # approx_params
        sha.update(json.dumps(self.ihvp_config, sort_keys=True).encode('utf-8'))
        return 'ihvp.' + sha.hexdigest() + '.npz'

    
    def _get_inverse_hvp_lissa(self, sess, test_grad_loss):
        ihvp_config = self.ihvp_config
        print_iter = ihvp_config['recursion_depth'] / 10

        inverse_hvp = None
        for _ in range(ihvp_config['num_repeats']):
            cur_estimate = test_grad_loss
            # debug_diffs_estimation = []
            # prev_estimation_norm = np.linalg.norm(np.concatenate([a.reshape(-1) for a in cur_estimate]))

            for j in range(ihvp_config['recursion_depth']):
                train_batch_data, train_batch_label = self.feeder.train_batch(ihvp_config['recursion_batch_size'])
                feed_dict = self._make_train_feed_dict(train_batch_data, train_batch_label)
                feed_dict = self._update_feed_dict(feed_dict, cur_estimate, test_grad_loss)

                if _using_fully_tf:
                    feed_dict.update({
                        self.v_param_damping: 1 - self.ihvp_config['damping'],
                        self.v_param_scale: self.ihvp_config['scale']
                    })
                    cur_estimate = sess.run(self.estimation_op, feed_dict=feed_dict)
                else:
                    hessian_vector_val = sess.run(self.hessian_vector_op, feed_dict=feed_dict)
                    hessian_vector_val = np.array(hessian_vector_val)
                    cur_estimate = test_grad_loss + (1 - ihvp_config['damping']) * cur_estimate - hessian_vector_val / ihvp_config['scale']

                if (j % print_iter == 0) or (j == ihvp_config['recursion_depth'] - 1):
                    logger.info("Recursion at depth %s: norm is %.8lf" %
                                (j, np.linalg.norm(np.concatenate([a.reshape(-1) for a in cur_estimate]))))

            if inverse_hvp is None:
                inverse_hvp = np.array(cur_estimate) / ihvp_config['scale']
            else:
                inverse_hvp += np.array(cur_estimate) / ihvp_config['scale']

            # np.savetxt(self._path('debug_diffs_estimation_{}.txt'.format(sample_idx)), debug_diffs_estimation)

        inverse_hvp /= ihvp_config['num_repeats']
        return inverse_hvp

    
    def _update_feed_dict(self, feed_dict, cur_estimated, test_grad_loss):
        for placeholder, var in zip(self.v_cur_estimated, cur_estimated):
            feed_dict[placeholder] = var

        for placeholder, var in zip(self.v_test_grad, test_grad_loss):
            feed_dict[placeholder] = var
        return feed_dict

    
    @_timing
    def _grad_diffs(self, sess, train_indices, num_total_train_example):
        inverse_hvp = np.concatenate([a.reshape(-1) for a in self.inverse_hvp])

        num_to_remove = len(train_indices)
        predicted_grad_diffs = np.zeros([num_to_remove])

        for counter, idx_to_remove in enumerate(train_indices):
            single_data, single_label = self.feeder.train_one(idx_to_remove)
            feed_dict = self._make_train_feed_dict([single_data], [single_label])
            predicted_grad_diffs[counter] = self._grad_diff(sess, feed_dict, num_total_train_example, inverse_hvp)

            if (counter % 1000) == 0:
                logger.info('counter: {} / {}'.format(counter, num_to_remove))

        return predicted_grad_diffs

    
    @_timing
    def _grad_diffs_all(self, sess, train_batch_size, num_iters, num_subsampling):
        num_total_train_example = num_iters * train_batch_size
        if num_subsampling > 0:
            num_diffs = num_iters * num_subsampling
        else:
            num_diffs = num_iters * train_batch_size

        inverse_hvp = np.concatenate([a.reshape(-1) for a in self.inverse_hvp])
        predicted_grad_diffs = np.zeros([num_diffs])

        counter = 0
        for it in range(num_iters):
            train_batch_data, train_batch_label = self.feeder.train_batch(train_batch_size)

            if num_subsampling > 0:
                for idx in range(num_subsampling):
                    feed_dict = self._make_train_feed_dict(train_batch_data[idx:idx + 1], train_batch_label[idx:idx + 1])
                    predicted_grad_diffs[counter] = self._grad_diff(sess, feed_dict, num_total_train_example, inverse_hvp)
                    counter += 1
            else:
                for single_data, single_label in zip(train_batch_data, train_batch_label):
                    feed_dict = self._make_train_feed_dict([single_data], [single_label])
                    predicted_grad_diffs[counter] = self._grad_diff(sess, feed_dict, num_total_train_example, inverse_hvp)
                    counter += 1

            if (it % 10) == 0:
                logger.info('iter: {}/{}'.format(it, num_iters))

        return predicted_grad_diffs

    
    def _grad_diff(self, sess, feed_dict, num_total_train_example, inverse_hvp):
        if _using_fully_tf:
            feed_dict.update({
                self.v_ihvp: inverse_hvp,
                self.v_param_total_trainset: num_total_train_example
            })
            return sess.run(self.grad_diff_op, feed_dict=feed_dict)
        else:
            train_grads = sess.run(self.grad_op_train, feed_dict=feed_dict)
            train_grads = np.concatenate([a.reshape(-1) for a in train_grads])
            train_grads /= num_total_train_example
            return np.dot(inverse_hvp, train_grads)

        
    def _make_test_feed_dict(self, xs, ys):
        ret = {
            self.x_placeholder: xs,
            self.y_placeholder: ys,
        }
        ret.update(self.test_feed_options)
        return ret

    
    def _make_train_feed_dict(self, xs, ys):
        ret = {
            self.x_placeholder: xs,
            self.y_placeholder: ys,
        }
        ret.update(self.train_feed_options)
        return ret

    
    def _path(self, *paths):
        return os.path.join(self.workspace, *paths)#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
import keras
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

from bokeh.io import output_notebook, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import brewer
from bokeh.models import (
  ColumnDataSource, ColorBar, 
  LinearColorMapper, LogColorMapper,
)
from bokeh.models.annotations import BoxAnnotation
from bokeh.models.tools import HoverTool
from bokeh.palettes import Viridis3, Viridis256, Category10, Plasma256
from bokeh.plotting import figure
from bokeh.transform import transform

import influence_debugger
from influence_debugger.influence.feeder import MNISTFeeder
from influence_debugger.influence.influence import Influence

# =============================================================================
# METHODS
# =============================================================================



# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================
def main(model):
    """Description of main()"""


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

    sorted_indices = np.argsort(scores)
    harmful = sorted_indices[:10]
    helpful = sorted_indices[-10:][::-1]

    print('\nHarmful:')
    for idx in harmful:
        print('[{}] {}'.format(idx, scores[idx]))

    print('\nHelpful:')
    for idx in helpful:
        print('[{}] {}'.format(idx, scores[idx]))

    fig, axes1 = plt.subplots(2, 5, figsize=(15, 5))
    target_idx = 0
    for j in range(2):
        for k in range(5):
            idx = helpful[target_idx]
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(feeder.train_origin_data[idx])
            label_str = int(feeder.train_origin_label[idx])
            axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))

            target_idx += 1

    fig, axes1 = plt.subplots(2, 5, figsize=(15, 5))
    target_idx = 0
    for j in range(2):
        for k in range(5):
            idx = harmful[target_idx]
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(feeder.train_origin_data[idx])
            label_str = int(feeder.train_origin_label[idx])
            axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))

            target_idx += 1

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
    plt.show()

    

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

    #@title Plotting functions for interactive visualization
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

    show(p)

    #@title Organizing indexes by rank

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

    show(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--model', '-m', type=str, help='path to .h5 file with model weights')
    parser.add_argument('--train_data', '-t', help='tensors for the training data')
    parser.add_argument('--test_data', '-r', help='tensors for the test dats')
    args = parser.parse_args()
    
    main()


