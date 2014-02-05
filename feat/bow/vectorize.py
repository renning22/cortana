# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 16:16:44 2014

@author: Ning
"""

import os,sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from util import *
from scipy.sparse import *
import cPickle as pickle
import codecs


def vectorize(tfidf=False,binary=False):
    
    log._logger.info("Loding...")
    
    trainX = [r[0] for r in tsv.reader(conv.redirect('train.tokenized.dat'))]
    testX = [r[0] for r in tsv.reader(conv.redirect('test.tokenized.dat'))]
    
    vectorizer = None
    if tfidf:
        vectorizer = TfidfVectorizer
    else:
        vectorizer = CountVectorizer
    
    log._logger.info("Fitting and transforming...")
    vectorizer = vectorizer(token_pattern=u'(?u)\\b\\w+\\b',binary=binary)
    trainX = vectorizer.fit_transform(trainX)
    testX = vectorizer.transform(testX)
    
    log._logger.info("Dumping...")
    pickle.dump(vectorizer,open("vectorizer.bin",'w'))
    pickle.dump(trainX,open("train.vectorized.mat",'w'))
    pickle.dump(testX,open("test.vectorized.mat",'w'))
    
    schema = vectorizer.get_feature_names()
    codecs.open("schema.dat",'w',encoding='utf-8').write('\n'.join(schema))

    # debug
    log._logger.info("Dumping inversered...")
    codecs.open("test.vectorized.dat",'w',encoding='utf-8').write( '\n'.join( [(' '.join(i)) for i in vectorizer.inverse_transform(testX)] ) )
    codecs.open("train.vectorized.dat",'w',encoding='utf-8').write( '\n'.join( [(' '.join(i)) for i in vectorizer.inverse_transform(trainX)] ) )
    
if __name__ == "__main__":
    vectorize()