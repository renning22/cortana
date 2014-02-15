# -*- coding: utf-8 -*-

import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from util.log import _logger
from util import *
from feat.terms.term_categorize import term_category


CLFs = {
    "nb": BernoulliNB(fit_prior = False),
    "sgd": SGDClassifier(penalty="l2", class_weight="auto", n_iter=100),
    "svm_ovr": LinearSVC(loss='l1', penalty="l2", multi_class="ovr", class_weight="auto"),
    "svm_sin": LinearSVC(loss='l1', penalty="l2", multi_class="crammer_singer"),
    "knn": KNeighborsClassifier(n_neighbors=10, weights = 'distance')
}

class Vectorizer(object):
    def __init__(self):
        self.count_vec = TfidfVectorizer(binary = True,
                                         ngram_range = (1, 3),
                                         tokenizer = Tokenizer())

        self.last_vec = CountVectorizer(binary = True, ngram_range = (1, 1), tokenizer = Tokenizer())


    def collect_last_term(self, X):
        X_last = list()
        tokens = self.last_vec.build_tokenizer()
        _logger.debug("Extracting last term for each sentence")
        for sent in X:
            X_last.append(tokens(sent)[-1])
        _logger.debug("Fitting last-term vectorizer")
        return X_last
        

    def fit(self, X, y = None):
        _logger.debug("Fitting count vectorizer")
        self.count_vec.fit(X)
        X_last = self.collect_last_term(X)
        self.last_vec.fit(X_last)
        return self

    def transform(self, X, y = None):
        #return self.count_vec.transform(X)
        _logger.debug("Doing tfidf transform")
        Xc = self.count_vec.transform(X)

        X_last = self.collect_last_term(X)
        _logger.debug("Doing last term transform")
        Xl = self.last_vec.transform(X_last)
        _logger.debug("stacking features")
        ret = sparse.hstack([Xc, Xl])
        
        tokens = self.count_vec.build_tokenizer()
        l = list()
        for sent in X:
            terms = tokens(sent)
            l.append(1 if  ("__LOCATION__" in terms and "__ORGNIZATION__" in terms) else 0)

        l = np.array(l)
        l.shape = len(l), 1
        ret = sparse.hstack([ret, l])
        _logger.debug("vectorization transform done")

        return ret


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data", default = TRAIN_FILE_PATH)
    cmd.add_argument("--algo", help="alogrithm to use", required=True, choices = CLFs.keys())
    args = cmd.parse_args()

    X, y = load_data(args.input)
    _logger.info("training using %s" % args.algo)

    pipeline = Pipeline([
            ("vert", TfidfVectorizer(min_df = 1, binary = True, ngram_range = (1, 3),
                                     tokenizer = Tokenizer())),
            #("vert", Vectorizer()),
            ("clf", CLFs[args.algo]),
            ])

    pipeline.fit(X, y)
    from decode import test
    test(TEST_FILE_PATH, pipeline)

    outpath = "%s.model" % args.algo
    with open(outpath, "w") as outfile:
        pickle.dump(pipeline, outfile)
        _logger.info("Model dumpped to %s" % outpath)
        



