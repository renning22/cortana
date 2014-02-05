# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:01:42 2014

@author: Ning
"""

from sklearn.svm import (SVC,LinearSVC)
import cPickle as pickle
from util import *
import codecs


def train(modelfile,trainmatfile):
    ""

    log._logger.info("Loading...")
    trainX = pickle.load(open(conv.redirect(trainmatfile)))
    trainy = [r[1] for r in tsv.reader(conv.redirect("data|train.dat"))]

    log._logger.info("Training...")
    clf = LinearSVC()
    clf.fit(trainX,trainy)
    
    log._logger.info("Dumping to %s" % (modelfile))
    
    pickle.dump(clf,open(modelfile,'w'))


def test(modelfile,testmatfile,outfile):
    ""
    
    clf = pickle.load(open(modelfile))
    
    testX = pickle.load(open(conv.redirect(testmatfile)))
    testy = [r[1] for r in tsv.reader(conv.redirect("data|test.dat"))]
    
    log._logger.info("Testing...")
    predicts = clf.predict(testX)
    
    with codecs.open(outfile,'w',encoding='utf-8') as fl:
        for src,p in zip(tsv.reader(conv.redirect("data|test.dat")),predicts):
            fl.write( "%s\t%s\t%s\n" % (src[0],src[1],p) )
    
if __name__ == "__main__":
    train("linear_svm.model","bow|train.vectorized.mat")
    test("linear_svm.model","bow|test.vectorized.mat","predicted.dat")
    