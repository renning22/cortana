# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:01:42 2014

1vs1 is almost as same performence as 1vsR. But dramatically more expensive time consuming
due to >= O(N^2) where N is number of trainning samples in libsvm's implementation

@author: Ning
"""

from sklearn.svm import (SVC,LinearSVC)
import cPickle as pickle
from util import *
import codecs


def linear_train(modelfile,trainmatfile,vs='1vsR',C=1,regularize='l1'):
    ""

    log._logger.info("linear_train : %s " % (modelfile))
    log._logger.info("Loading...")
    trainX = pickle.load(open(conv.redirect(trainmatfile)))
    trainy = [r[1] for r in tsv.reader(conv.redirect("data|train.dat"))]

    # Optimation
    trainX = trainX.tocsr(False)
    
    log._logger.info("Training...")
    if vs == '1vsR':
        if regularize == 'l1':
            clf = LinearSVC(loss='l2',penalty='l1',dual=False)
        else:
            clf = LinearSVC(loss='l1',penalty='l2',dual=True)
    elif vs == '1vs1':
        clf = SVC(kernel='linear')
    else:
        raise "Not supported"
        
    clf.fit(trainX,trainy)
    
    log._logger.info("Dumping to %s" % (modelfile))
    
    pickle.dump(clf,open(modelfile,'w'))
    
    return clf


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
    #linear_train("linear_1vs1_l1.model","bow|train.vectorized.mat",vs='1vs1',regularize='l1')
    #test("linear_1vs1_l1.model","bow|test.vectorized.mat","linear_1vs1_l1.predicted.dat")

    linear_train("linear_1vsR_l1.model","bow|train.vectorized.mat",vs='1vsR',regularize='l1')
    test("linear_1vsR_l1.model","bow|test.vectorized.mat","linear_1vsR_l1.predicted.dat")
    