import os, sys, math
import cPickle as pickle
import argparse
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from util.log import _logger
from util import *

TRIGGER_THRESHOLD = 2.0

TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../../raw_data/aggregated/test.dat")
front = None

def test(X, y):
    _logger.info("Fisrt stage accuracy: %f" % front.score(X, y))
    import decode_svm
    outfile = open("predicted.dat", "w")
    discfile = open("discriminated.dat", "w")
    y_pred = list()
    sz = len(y)
    domains = front.named_steps["clf"].classes_
    for i in xrange(sz):
        sent = X[i]
        gold = y[i]
        
        front_result = sorted(zip(domains, front.decision_function([sent])[0]),
                              key = lambda x: -x[1])
        
        pred = front_result[0][0]
        assert pred == front.predict([sent])[0]

        if front_result[0][1] - front_result[1][1] < TRIGGER_THRESHOLD:
            p = front_result[0][0]
            q = front_result[1][0]
            svm_pred = decode_svm.discriminate(p, q, sent)[0]
            discfile.write("%s\t%s\t%s\t%s\n" % \
                               (sent.encode('utf-8'), pred.encode('utf-8'),
                                svm_pred.encode('utf-8'), gold.encode('utf-8')))
            pred = svm_pred

        y_pred.append(pred)

        outfile.write("%s\t%s\t%s\n" % (sent.encode('utf-8'), pred.encode('utf-8'), gold.encode('utf-8')))

    _logger.info("ensembled accuracy: %f" % accuracy_score(y, y_pred))

    outfile.close()
    discfile.close()

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--serv", help = "run as server", dest="as_server", action='store_true')
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    cmd.add_argument("--front-model-path", help = "path to the first stage model")
    args = cmd.parse_args()

    _logger.info("Loading naive bayes model from %s" % args.front_model_path)
    front = pickle.load(open(args.front_model_path))
    X, y = load_data(args.path)
    test(X, y)

