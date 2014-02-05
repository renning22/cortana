# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:26:25 2014

@author: Ning
"""

import csv

def reader(filename):
    for row in csv.reader(open(filename,'r'),delimiter='\t'):
        ret = []
        for r in row:
            ret.append( unicode(r,'utf-8') )
        yield ret