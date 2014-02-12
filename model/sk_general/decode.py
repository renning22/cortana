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

from train import CLFs

def serv(clf):
    domains = clf.named_steps['clf'].classes_
    while True:
        query = raw_input('Input your query(must be segmented by SPACE), q to quit:\n').decode('utf-8')
        if query == u'q':
            return
        detail = sorted(zip(domains, clf.decision_function([query])[0]),
                        key = lambda x: -x[1])
        print 'result:', clf.predict([query])[0], '\n'
        for domain, val in detail:
            print domain, val

if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    cmd.add_argument("--serv", help = "run as server", dest="as_server", action='store_true')
    cmd.add_argument("--model", help = "path to the pickled model", required=True,
                     choices = ["%s.model" % algo for algo in CLFs.keys()])
    args = cmd.parse_args()

    _logger.info("loading model from %s" % args.model)
    clf = pickle.load(open(args.model))

    if args.as_server:
        serv(clf)
        sys.exit(0)

    X, y = load_data(args.path)
    
    y_pred = clf.predict(X)
    outfile = open("%s.predicted.dat" % args.model.split('.')[0], 'w')
    for i in range(len(y)):
        sentence, pred, gold = X[i], y_pred[i], y[i]
        outfile.write("%s\t%s\t%s\n" % (sentence.encode('utf-8'), pred.encode('utf-8'), gold.encode('utf-8')))

    _logger.info("accuracy: %f, %d records" % (accuracy_score(y, y_pred),
                                               len(y)))
