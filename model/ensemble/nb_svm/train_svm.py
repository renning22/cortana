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
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from util.log import _logger
from util import *
from feat.terms.term_categorize import term_category

class SVMGroup(object):
    def __init__(self, cv=0):
        self.cv = cv
        self.svms = defaultdict(list)
        self.by_domain_data = defaultdict(list)

    def collect_by_domain(self, X, y):
        sz = len(y)
        self.by_domain_data = defaultdict(list)
        for i in xrange(sz):
            domain = y[i]
            self.by_domain_data[domain].append(X[i])

    def train(self, X, y):
        self.svms = dict()
        self.domains = list(set(y))
        self.collect_by_domain(X, y)

        for p in self.domains:
            for q in self.domains:
                if p < q:
                    self.svms[p, q] = self.train_pair(p, q)

    def train_pair(self, p, q):
        if p > q:
            p, q = q, p

        p_len = len(self.by_domain_data[p])
        q_len = len(self.by_domain_data[q])

        _logger.info("Training SVM for %s V.S. %s, %d + %d = %d recored" % \
                         (p, q, p_len, q_len, p_len + q_len))

        X = list(self.by_domain_data[p])
        X.extend(self.by_domain_data[q])
        y = [p] * p_len
        y.extend([q] * q_len)

        pipeline = Pipeline([
                ("vert", TfidfVectorizer(min_df = 1, binary = False, ngram_range = (1, 1),
                                         tokenizer = Tokenizer())),
                ("svm", LinearSVC(loss='l2', penalty="l1",
                                  dual=False, tol=1e-3)),
                ])

        if self.cv > 0:
            _logger.info("Doing grid search on %d fold CV" % self.cv)
            params = {
                "svm__C": [1, 10, 50, 100, 500, 1000],
                }
            grid = GridSearchCV(pipeline, params, cv=self.cv, verbose=50)
            grid.fit(X, y)
            pipeline = grid.best_estimator_
            _logger.info("Grid search got best score:%f" % grid.best_score_)
            pipeline.accur = grid.best_score_
        else:
            pipeline.fit(X, y)
            _logger.debug("Testing on training data")
            accur = accuracy_score(y, pipeline.predict(X))
            pipeline.accur = accur
            _logger.info("Trainig accuracy (%s - %s): %f" % (p, q, accur))
        self.svms[p,q] = pipeline
        return pipeline
                    

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data", default=TRAIN_FILE_PATH)
    cmd.add_argument("--classes", help="the pair of classes need to train, train all combination if not specified",
                     nargs=2, default=None)
    cmd.add_argument("--cv", help="fold of cross validation 0 for not doing", default=0, type=int)
    args = cmd.parse_args()

    _logger.info("Loading training data from %s" % args.input)
    X, y = load_data(args.input)

    if args.classes:
        _logger.info("Will train 1v1 SVM between %s and %s" % (args.classes[0], args.classes[1]))
        gp = SVMGroup(cv=args.cv)
        if os.path.isfile("svms.model"):
            gp.svms = pickle.load(open("svms.model"))

        gp.collect_by_domain(X, y)
        gp.train_pair(args.classes[0], args.classes[1])

    else:
        gp = SVMGroup()
        _logger.info("Start training")
        gp.train(X, y)

    with open("svms.model", "w") as outfile:
        pickle.dump(gp.svms, outfile)
        _logger.info("SVM models dumped to svms.model")

