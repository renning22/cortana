import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
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
from featurized.terms.term_categorize import term_category

CLFs = {
    "nb": BernoulliNB(fit_prior = False),
    "sgd": SGDClassifier(alpha=.0001, n_iter=50,
                         penalty="l2"),
    "svm": LinearSVC(loss='l1', penalty="l2"),
    "knn": KNeighborsClassifier(n_neighbors=10, weights = 'distance')
}



if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data", required=True)
    cmd.add_argument("--algo", help="alogrithm to use", required=True, choices = CLFs.keys())
    args = cmd.parse_args()

    X, y = load_data(args.input)
    _logger.info("Will use algorithm %s" % args.algo)

    pipeline = Pipeline([
            ("vert", TfidfVectorizer(min_df = 1, binary = False, ngram_range = (1, 1), analyzer = Analyzer())),
            #("select", SelectKBest(chi2, k=12000)),
            ("clf", CLFs[args.algo]),
            ])


    pipeline.fit(X, y)
    outpath = "%s.model" % args.algo
    with open(outpath, "w") as outfile:
        pickle.dump(pipeline, outfile)
        _logger.info("Model dumpped to %s" % outpath)
        



