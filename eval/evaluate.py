# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 02:30:16 2014

@author: Ning
"""

import pandas
import numpy as np
from collections import defaultdict
from util import *
import codecs

def evaluate(predictfile,outfile):
    
    classes = { j[0]:i for i,j in enumerate(tsv.reader(conv.redirect("data|class.dat"))) }
    
    table = np.zeros(shape=(len(classes),len(classes)), dtype=int)
    
    for i in tsv.reader(predictfile):
        table[ classes[i[1]] , classes[i[2]] ] += 1
    
    correct = table.trace()
    total = table.sum()
    
    accuracy = float(correct)/float(total)
    
    with codecs.open(outfile,'w',encoding='utf-8') as fl:
        fl.write( "Total = %s\n" % total )
        fl.write( "Correct = %s\n" % correct )
        fl.write( "Accuracy = %s\n" % accuracy )
        fl.write( "\n" )
        
        fl.write( "Confusion Matrix:\n" )
        
        frame = pandas.DataFrame(table,sorted(classes.keys()),sorted(classes.keys()))
        fl.write( "%s\n"%(str(frame)) )
            
if __name__ == "__main__":
    evaluate(conv.redirect("svm|predicted.dat"),"svm_results.dat")
    evaluate(conv.redirect("naive|predicted.dat"),"naive_results.dat")