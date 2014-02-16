# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:01:42 2014

1vs1 is almost as same performence as 1vsR. But dramatically more expensive time consuming
due to >= O(N^2) where N is number of trainning samples in libsvm's implementation

@author: Ning
"""

from sklearn.svm import (SVC,LinearSVC)
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
from util import *
import codecs
import argparse
from scipy.sparse import csr_matrix

def linear_train(trainfile,testfile,vs='1vsR',C=1,reg='l2', dump=False):
    ""

    log._logger.info("linear_train : %s , %s" % (trainfile,testfile))
    log._logger.info("Loading...")

    trainX = pickle.load(open(trainfile))
    trainy = [r[1] for r in tsv.reader(conv.redirect("data|train.dat"))]

    # Normalize
    #log._logger.info("Normolizing...")
    scaler = StandardScaler(copy=True,with_mean=False)
    #log._logger.info( str(scaler) )
    #trainX = scaler.fit_transform(trainX)

    # Optimation
    # trainX = csr_matrix(trainX)
    
    log._logger.info("Training...")
    if vs == '1vsR':
        if reg == 'l1':
            clf = LinearSVC(loss='l2',penalty='l1',dual=False,C=C,class_weight='auto')
        else:
            clf = LinearSVC(loss='l1',penalty='l2',dual=True,C=C,class_weight='auto')
    elif vs == '1vs1':
        clf = SVC(kernel='linear')
    else:
        raise "Not supported"

    print str(clf)+'\n'
        
    clf.fit(trainX,trainy)

    if dump:
        log._logger.info("Dumping to %s" % ('svm.model'))
        pickle.dump(clf,open("svm.model",'w'))
    
    if testfile is not None:
        test(clf,scaler,testfile)

    return clf


def test(model,scaler,testmatfile):
    ""
    
    clf = model
    
    testX = pickle.load(open(testmatfile))
    testy = [r[1] for r in tsv.reader(conv.redirect("data|test.dat"))]
    
    #log._logger.info("Normalizing...")
    #testX = scaler.transform(testX)

    log._logger.info("Testing...")
    predicts = clf.predict(testX)
    
    with codecs.open("svm.predicted.dat",'w',encoding='utf-8') as fl:
        for src,p in zip(tsv.reader(conv.redirect("data|test.dat")),predicts):
            fl.write( "%s\t%s\t%s\n" % (src[0],p, src[1]) )
    
    
if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="which feature you use",default="com")
    cmd.add_argument("--reg", help="reg",default="l2")
    cmd.add_argument("--C", help="parameter C", type=float, default=1)
    cmd.add_argument("--vs", help="enable vs", default='1vsR')
    
    args = cmd.parse_args()

    train_feature_file = conv.redirect(args.input+"|train.vectorized.mat")
    test_feature_file = conv.redirect(args.input+"|test.vectorized.mat")
    
    linear_train(trainfile=train_feature_file,testfile=test_feature_file,vs=args.vs,reg=args.reg,C=args.C)
    
