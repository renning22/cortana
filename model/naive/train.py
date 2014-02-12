# coding=utf-8

# This is a most naive algorithm. For each sentence it assumes each word is generated by the
# domain it belongs to. It simply predict the target domain d as the one that maximize the value of:
# P(w1|d) * P(w2|d) * ... * P(wn|d)
# Use backoff discounting for words never occured in domain d. Ignore words never seen in the 
# training data.

import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
from sklearn import cross_validation
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util.log import _logger
from featurized.terms.term_categorize import term_category

class NaiveBayes(object):
    def __init__(self, train_path, term_path, alpha = 0.5):
        self.train_path = train_path
        self.term_path = term_path
        self.alpha = alpha
        self.reset()
        self.load_data()

    def get_category(self, term):
        return term_category(term)

    def load_data(self):
        _logger.info("Loading training data from %s" % self.train_path)
        self.X = []
        self.y = []
        with open(self.train_path) as train_file:
            for line in train_file:
                line = line.strip().decode('utf-8')
                if not line:
                    continue
                terms, domain = line.split('\t')
                self.X.append(terms)
                self.y.append(domain)

    def reset(self):
        self.training_sentence_count = 0
        self.count = defaultdict(int)
        self.domain_has = defaultdict(set)
        self.domain_count = defaultdict(int) # Number of sentence in each domain
        self.domain_backoff = dict()
        self.term_count = defaultdict(int)
        self.terms = set()

    def fit(self, X, y):
        self.reset()
        size = len(y)
        for i in xrange(size):
            if (i + 1) % 10000 == 0:
                _logger.debug("%d processed" % (i+1))
            terms = X[i]
            domain = y[i]
            self.training_sentence_count += 1
            terms = terms.split(' ')
            self.domain_count[domain] += 1
            term_set = set()
            for term in terms:
                term = self.get_category(term)
                if term in term_set:
                    continue
                term_set.add(term)
                self.terms.add(term)
                self.count[term, domain] += 1
                self.count[domain] += 1
                self.term_count[term] += 1
                self.domain_has[domain].add(term)

        for domain in self.domain_has:
            backoff = len(self.domain_has[domain]) * self.alpha / self.count[domain]
            backoff /= len(self.term_count) - len(self.domain_has[domain])
            self.domain_backoff[domain] = backoff

        self.domains = self.domain_backoff.keys()

    def train(self):
        self.fit(self.X, self.y)

    def get_test_accuracy(self, X, y):
        total = len(y)
        correct = 0.0
        from decode import NaiveDecoder
        decoder = NaiveDecoder(self)
        for idx, x in enumerate(X):
            y_pred = decoder.predict(x)
            if y_pred == y[idx]:
                correct += 1
        return correct / total

    def cv(self, fold):
        size = len(self.y)
        kf = cross_validation.KFold(size, fold, shuffle=True)
        iteration = 0
        scores = list()
        for train_idx, test_idx in kf:
            X = [self.X[idx] for idx in train_idx]
            y = [self.y[idx] for idx in train_idx]
            X_test = [self.X[idx] for idx in test_idx]
            y_test = [self.y[idx] for idx in test_idx]
            _logger.debug("Training...")
            self.fit(X, y)
            _logger.debug("Testing...")
            score = self.get_test_accuracy(X_test, y_test)
            scores.append(score)
            iteration += 1
            _logger.info("CV iteration %d: CV accuracy: %f" % \
                             (iteration, score))

        scores = np.array(scores)
        return scores.mean(), scores.std()

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data")
    cmd.add_argument("--terms", help="path of the terms file")
    cmd.add_argument("--alpha", help="alpha of discounting", type=float, default=0.5)
    cmd.add_argument("--cv", help="enable cross validation", type=int, default=0)
    args = cmd.parse_args()

    naive = NaiveBayes(args.input, args.terms, args.alpha)
    if args.cv > 0:
        _logger.info("CV accuracy: %f +/- %f" % naive.cv(args.cv))
    else:
        _logger.info("Start training");
        naive.train()
        with open("naive.model", "w") as outfile:
            pickle.dump(naive, outfile)
            _logger.info("Model dumped to naive.model")
