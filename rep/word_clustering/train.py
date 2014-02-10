import os, sys, math, argparse
import cPickle as pickle
import numpy as np
from sklearn.cluster import KMeans
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util import *
from util.log import _logger
from featurized.terms.term_categorize import term_category, g_term_count
from model.naive.train import NaiveBayes
from model.naive.decode import NaiveDecoder

class Clustering(object):
    def __init__(self, naive_model_path):
        _logger.info("loading naive bayes model from %s" % naive_model_path)
        model = pickle.load(open(naive_model_path))
        self.naive = NaiveDecoder(model)
        self.words = dict()
    
    def train(self):
        _logger.info("reading posterior probabilities from naive bayes model")
        self.words = list()
        self.words_seen = set()
        X = np.array([])
        for term in g_term_count:
            term = term_category(term)
            if term in self.words_seen:
                continue
            self.words_seen.add(term)
            self.words.append(term)
            x = list()
            for domain in self.naive.model.domains:
                val = self.naive.posterior_prob(term, domain)
                x.append(val)
            X = np.append(X, x)
        _logger.info("%d terms need to be clustered" % len(self.words))

        X = np.reshape(X, (len(self.words), len(self.naive.model.domains)))
        kmeans = KMeans(n_clusters = len(self.words) / 10)
        y = kmeans.fit_predict(X)

        with open(OUTFILE_PATH, "w") as outfile:
            for i in xrange(len(y)):
                outfile.write("%s\t%d\n" % (self.words[i].encode('utf-8'), y[i]))
        _logger.info("clustering result wrote to %s" % OUTFILE_PATH)
            

OUTFILE_PATH = "./clusters.txt"
NAIVE_BAYES_MODEL_PATH = ROOT + "/model/naive/naive.model"

if __name__ == "__main__":
    clustering = Clustering(NAIVE_BAYES_MODEL_PATH)
    clustering.train()
