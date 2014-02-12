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

svms = None

def discriminate(p, q, sent, detail = False):
    key = tuple(sorted([p, q]))
    clf = svms[key]
    # _logger.debug("comparing between %s and %s" % (clf.named_steps['svm'].classes_[0],
    #                                                clf.named_steps['svm'].classes_[1]))
    if detail:
        return clf.predict([sent]), clf.decision_function([sent])[0]

    return clf.predict([sent])

def serv():
    while True:
        query = raw_input('Input your query(must be segmented by SPACE), q to quit:\n').decode('utf-8')
        if query == u'q':
            break
        domains = raw_input('Input the domains you want to compare:\n').decode('utf-8')
        domains = domains.split()
        if len(domains) != 2:
            print "must have two domains:", len(domains), domains
            continue

        print discriminate(domains[0], domains[1], query, detail = True)

def test(X, y):
    by_domain = defaultdict(list)
    sz = len(y)
    for i in xrange(sz):
        by_domain[y[i]].append(X[i])

    domains = ['alarm', 'calendar', 'communication', 'note', 'places',
               'reminder', 'weather', 'web']
    for p in domains:
        for q in domains:
            if p < q:
                clf = svms[p, q]
                p_len = len(by_domain[p])
                q_len = len(by_domain[q])
                X = list(by_domain[p])
                X.extend(by_domain[q])
                y = [p] * p_len
                y.extend([q] * q_len)
                _logger.info("%.4f, %s - %s" % (clf.score(X, y), p, q))


_logger.info("loading model from svms.model")
svms = pickle.load(open('svms.model'))
        
if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    cmd.add_argument("--serv", help = "run as server", dest="as_server", action='store_true')
    args = cmd.parse_args()
    X, y = load_data(args.path)

    if args.as_server:
        serv()
    else:
        test(X, y)
