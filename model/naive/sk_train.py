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
from sklearn.feature_extraction.text import CountVectorizer
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util.log import _logger
from featurized.terms.term_categorize import term_category
from rep.gini.decode import top as top_gini

class Analyzer(object):
    def __init__(self):
        self.top_gini = set(top_gini(4000))

    def __call__(self, sentence):
        terms = sentence.strip().split(' ')
        ret = set()
        for term in terms:
            term = term_category(term)
            if term in self.top_gini:
                ret.add(term)
            
        return list(ret)
        

pipeline = Pipeline([
        ("vert", CountVectorizer(min_df = 1, binary = True, ngram_range = (1, 1), analyzer = Analyzer())),
        #("nb", BernoulliNB(fit_prior = False)),
        ("logreg", LogisticRegression())
        ])

params = {
    #"nb__alpha": [0.1, 1, 10],
    "logreg__penalty": ["l1", "l2"],
    "logreg__C": [0.01, 0.1, 1, 10, 100]
    }

def load_data(train_path):
    _logger.info("Loading training data from %s" % train_path)
    X = []
    y = []
    with open(train_path) as train_file:
        for line in train_file:
            line = line.strip().decode('utf-8')
            if not line:
                continue
            terms, domain = line.split('\t')
            X.append(terms)
            y.append(domain)
    return X, y




if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data")
    cmd.add_argument("--cv", help="enable cross validation", type=int, default=0)
    args = cmd.parse_args()

    X, y = load_data(args.input)

    if args.cv > 0:
        _logger.info("Doing %d fold cross validation" % args.cv)
        gs = GridSearchCV(pipeline, params, cv = args.cv, verbose=5)
        gs.fit(X, y)
        print gs._best_estimator_
        print gs._best_score_
        with open("sk_naive.model", "w") as outfile:
            pickle.dump(gs._best_score_, outfile)
            _logger.info("Model dumped to sk_naive.model")
        
        
    else:
        _logger.info("Start training")
        pipeline.fit(X, y)
        with open("sk_naive.model", "w") as outfile:
            pickle.dump(pipeline, outfile)
            _logger.info("Model dumped to sk_naive.model")
