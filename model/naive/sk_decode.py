import sys, os
import argparse
import cPickle as pickle
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util.log import _logger
from model.naive.sk_train import load_data, Analyzer

TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../raw_data/aggregated/test.dat")

if __name__ == "__main__":
    _logger.info("loading model")
    clf = pickle.load(open('sk_naive.model'))
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    args = cmd.parse_args()

    X, y = load_data(args.path)
    y_pred = clf.predict(X)
    outfile = open("predicted.dat", 'w')
    for i in range(len(y)):
        sentence, pred, gold = X[i], y_pred[i], y[i]
        outfile.write("%s\t%s\t%s\n" % (sentence.encode('utf-8'), pred.encode('utf-8'), gold.encode('utf-8')))
    

