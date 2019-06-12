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
:ORGANIZATION: UnifyID
:CONTACT: natthew@unify.id
:SINCE: Sat Jun  8 14:43:35 2019
:VERSION: 0.1
"""
# =============================================================================
# PROGRAM METADATA
# =============================================================================
__author__ = 'Matthew McAteer'
__contact__ = 'matthewmcateer0@gmail.com'
__copyright__ = 'UnifyID'
__license__ = ''
__date__ = 'Sat Jun  8 14:43:35 2019'
__version__ = '0.1'

# =============================================================================
# IMPORT STATEMENTS
# =============================================================================
import logging
from logging.handlers import RotatingFileHandler

# =============================================================================
# METHODS
# =============================================================================

class DebuggingLogger:
    """ logger fior debugging tools
    """
    def __init__(self):
        _formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-4s: %(message)s')
        # _file_handler = logging.handler.FileHandler(__package__ + '.log')
        _file_handler = RotatingFileHandler('debugging.log', maxBytes=1024*1024*100)
        _file_handler.setFormatter(_formatter)
        _file_handler.setLevel(logging.DEBUG)
        _stream_handler = logging.StreamHandler()
        _stream_handler.setFormatter(_formatter)
        _stream_handler.setLevel(logging.INFO)

        _logger = logging.getLogger(__package__)
        _logger.setLevel(logging.DEBUG)
        _logger.addHandler(_file_handler)
        _logger.addHandler(_stream_handler)
        self._logger = _logger

        _logger.debug('----------------------------')
        _logger.debug('start logging debugging package')

    @property
    def logger(self):
        return self._logger


logger = DebuggingLogger().logger
