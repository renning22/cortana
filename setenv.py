# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 17:10:20 2014

Setup a debug environment

Remember run this script firstly when you start a new python interpreter.

@author: Ning
"""

import sys,os

root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)