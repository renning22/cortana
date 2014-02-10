import os, sys, math
import cPickle as pickle
import argparse
import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util import *
from util.log import _logger
from model.naive.train import NaiveBayes
from featurized.terms.term_categorize import term_category, g_term_count
import rep.word_clustering.decode as word_clustering

class ClusteredNaiveBayes(NaiveBayes):
    def get_category(self, term):
        term = term_category(term)
        return word_clustering.get_cluster(term)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data")
    cmd.add_argument("--terms", help="path of the terms file")
    cmd.add_argument("--alpha", help="alpha of discounting", type=float, default=0.5)
    cmd.add_argument("--cv", help="enable cross validation", type=int, default=0)

    args = cmd.parse_args()

    naive = ClusteredNaiveBayes(args.input, args.terms, args.alpha)
    if args.cv > 0:
        _logger.info("CV accuracy: %f +/- %f" % naive.cv(args.cv))
    else:
        _logger.info("Start training");
        naive.train()
        with open("naive.clustered.model", "w") as outfile:
            pickle.dump(naive, outfile)
            _logger.info("Model dumped to naive.clustered.model")
