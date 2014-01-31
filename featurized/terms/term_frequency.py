# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:47:59 2014

@author: niren
"""

import sys,os,re
import csv

def term_frequency():
    
    out_dic = dict()
    
    path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + "/../../raw_data/aggregated/train.dat"
    
    with open(path,"r") as lines:
        tsvin = csv.reader(lines,delimiter='\t')
        for row in tsvin:
            for term in row[0].split():
                if not out_dic.has_key(term):
                    out_dic[term] = int(0);
                out_dic[term] = out_dic[term] + 1;
    
    with open("terms.dat",'w') as fl:
        for a, b in out_dic.items():
            fl.write( "%s\t%s\n" % (a,b) )
            
term_frequency()