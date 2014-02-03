# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 16:16:44 2014

@author: Ning
"""

import os,sys
from sklearn.feature_extraction.text import CountVectorizer
from util import *

vectorizer = CountVectorizer(min_df=1)



if __name__ == "__main__":
    print root