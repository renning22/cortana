# -*- coding: utf-8 -*-

import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from util.log import _logger
from util import *
from feat.terms.term_categorize import term_category

def clean(X, y, k=10):
    _logger.info("cleaning base on %d-fold cross validation" % k)

    size = len(y)
    kf = KFold(size, n_folds=k, shuffle=True)
    fold = 1
    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline = Pipeline([
                ("vert", TfidfVectorizer(min_df = 1, binary = True, ngram_range = (1, 3),
                                         tokenizer = Tokenizer())),
                ("clf", LinearSVC(loss='l1',
                                  penalty="l2",
                                  multi_class="ovr",
                                  class_weight="auto")),
                ])
        _logger.debug("Training fold %d" % fold)
        pipeline.fit(X_train, y_train)
        _logger.debug("Predicting for fold %d" % fold)
        y_pred = pipeline.predict(X_test)
        _logger.info("fold %d got accuracy: %f" % (fold, accuracy_score(y_test, y_pred)))

        right_f = open("fold%d.right.dat" % fold, "w")
        wrong_f = open("fold%d.wrong.dat" % fold, "w")

        size = len(y_test)
        for i in xrange(size):
            sent, pred, gold = X_test[i].encode('utf-8'), y_pred[i].encode('utf-8'), y_test[i].encode('utf-8')
            if pred != gold:
                wrong_f.write("%s\t%s\t%s\n" % (pred, gold, sent))
            else:
                right_f.write("%s\t%s\n" % (sent, gold))

        right_f.close()
        wrong_f.close()

        fold +=1

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data", default = TRAIN_FILE_PATH)
    cmd.add_argument("--fold", help="number of fold", default = 10, type = int)
    args = cmd.parse_args()
    k = args.fold

    X, y = load_data(args.input)
    clean(X, y, k)
