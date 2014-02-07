# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 02:30:16 2014

@author: Ning
"""

import pandas
import numpy as np
from util import *
import codecs
from cStringIO import StringIO

def evaluate(predictfile,outfile):
    
    classes = { j[0]:i for i,j in enumerate(tsv.reader(conv.redirect("data|class.dat"))) }
    
    table = np.zeros(shape=(len(classes),len(classes)), dtype=int)
    
    for i in tsv.reader(predictfile):
        table[ classes[i[1]] , classes[i[2]] ] += 1
    
    correct = table.trace()
    total = table.sum()
    
    accuracy = float(correct)/float(total)
    
    output = StringIO()
    output.write( "Total = %s\n" % total )
    output.write( "Correct = %s\n" % correct )
    output.write( "Accuracy = %s\n" % accuracy )
    output.write( "\n" )
    
    output.write( "Confusion Matrix:\n" )
    
    frame = pandas.DataFrame(table,sorted(classes.keys()),sorted(classes.keys()))
    output.write( "%s\n"%(str(frame)) )
    
    print output.getvalue()
    
    with codecs.open(outfile,'w',encoding='utf-8') as fl:
        fl.write(output.getvalue())
            
if __name__ == "__main__":
    evaluate(conv.redirect("svm|predicted.dat"),"svm_results.dat")
    evaluate(conv.redirect("naive|predicted.dat"),"naive_results.dat")