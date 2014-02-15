import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from util.log import _logger
from util import *
from feat.terms.term_categorize import term_category
from rep.gini.decode import top as top_gini
        

pipeline = Pipeline([
        ("vert", CountVectorizer(binary = True, ngram_range = (1, 3), tokenizer = Tokenizer())),
        ("nb", BernoulliNB(fit_prior = False)),
        ])

params = {
    "nb__alpha": [0.001, 0.01, 0.1, 0.5],
    }

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data", default=TRAIN_FILE_PATH)
    cmd.add_argument("--cv", help="enable cross validation", type=int, default=0)
    args = cmd.parse_args()

    X, y = load_data(args.input)

    if args.cv > 0:
        _logger.info("Doing %d fold cross validation" % args.cv)
        gs = GridSearchCV(pipeline, params, cv = args.cv, verbose=5)
        gs.fit(X, y)

        with open("sk_naive.model", "w") as outfile:
            pickle.dump(gs.best_estimator_, outfile)
            _logger.info("Model dumped to sk_naive.model")        
        print gs.best_estimator_
        print gs.best_score_
    else:
        _logger.info("Start training")
        pipeline.fit(X, y)
        with open("sk_naive.model", "w") as outfile:
            pickle.dump(pipeline, outfile)
            _logger.info("Model dumped to sk_naive.model")
