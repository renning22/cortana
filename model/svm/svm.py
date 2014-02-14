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
import argparse
from scipy.sparse import csr_matrix

def linear_train(modelfile,trainmatfile,vs='1vsR',C=1,regularize='l2',gridsearch=False):
    ""

    log._logger.info("linear_train : %s " % (modelfile))
    log._logger.info("Loading...")

    trainX = pickle.load(open(trainmatfile))
    trainy = [r[1] for r in tsv.reader(conv.redirect("data|train.dat"))]

    # Optimation
    trainX = csr_matrix(trainX)
    
    log._logger.info("Training...")
    if vs == '1vsR':
        if regularize == 'l1':
            clf = LinearSVC(loss='l2',penalty='l1',dual=False,C=C)
        else:
            clf = LinearSVC(loss='l1',penalty='l2',dual=True,C=C)
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
    
    testX = pickle.load(open(testmatfile))
    testy = [r[1] for r in tsv.reader(conv.redirect("data|test.dat"))]
    
    log._logger.info("Testing...")
    predicts = clf.predict(testX)
    
    with codecs.open(outfile,'w',encoding='utf-8') as fl:
        for src,p in zip(tsv.reader(conv.redirect("data|test.dat")),predicts):
            fl.write( "%s\t%s\t%s\n" % (src[0],p, src[1]) )
    
if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="which feature you use",default="bow")
    cmd.add_argument("--output", help="path of the output",default="svm.predicted.dat")
    cmd.add_argument("--regularize", help="regularition",default="l2")
    cmd.add_argument("--C", help="alpha of discounting", type=float, default=1)
    cmd.add_argument("--vs", help="enable vs", default='1vsR')
    cmd.add_argument("--gridsearch", help="enable gridsearch", type=bool, default=False)
    
    args = cmd.parse_args()

        
    linear_train("svm.model",trainmatfile=conv.redirect(args.input+"|train.vectorized.mat"),vs=args.vs,regularize=args.regularize,gridsearch=args.gridsearch,C=args.C)
    test("svm.model",conv.redirect(args.input+"|test.vectorized.mat"),args.output)
    
