#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

"""
LIME view for model debugging

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
import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import resize as skimage_resize

from .guided_grad import replace_grad_to_guided_grad
from .candidate_ops import candidate_featuremap_op_names, candidate_predict_op_names
from .candidate_ops import _unusable_ops

# =============================================================================
# METHODS
# =============================================================================









