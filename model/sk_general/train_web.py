import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from util.log import _logger
from util import *
from feat.terms.term_categorize import term_category

from train import Vectorizer


def gen(path, clf):
    X, y = load_data(path)
    scores = clf.decision_function(X)
    sz = len(y)
    with open('web_split.dat', 'w') as outfile:
        for i in xrange(sz):
            assert y[i] == 'web'
            score = scores[i]
            detail = sorted(zip(clf.named_steps['clf'].classes_,
                                score),
                            key = lambda x: -x[1])
            outfile.write('%s %f\n' % (detail[0][0], detail[0][1]))

if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--path", help = "path to only-web training data")
    cmd.add_argument("--serv", help = "run as server", dest="as_server", action='store_true')
    cmd.add_argument("--gen", help = "generate training data", dest="generate", action='store_true')

    args = cmd.parse_args()

    _logger.info("loading model from %s" % 'svm_ovr.model')
    clf = pickle.load(open('svm_ovr.model'))

    if args.generate:
        gen(args.path, clf)

