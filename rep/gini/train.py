# coding=utf-8

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

MODEL_PATH = './gini.dat'

class GiniCoe(object):
    def __init__(self, train_path):
        self.train_path = train_path
    
    def train(self):
        self.count = defaultdict(int)
        self.terms = set()
        self.domains = set()
        self.gini = dict()
        with open(self.train_path) as infile:
            for line in infile:
                term, domain, count = line.split(' ')
                count = int(count)
                self.count[term, domain] = count
                self.count[domain] += count
                self.terms.add(term)
                self.domains.add(domain)

        v = len(self.terms)
        for term in self.terms:
            p = dict()
            for domain in self.domains:
                p[domain] = (1.0 + self.count[term, domain]) / (v + self.count[domain])
            wcp = dict()
            s = sum(p.values())
            for domain in self.domains:
                wcp[domain] = p[domain] / s
            self.gini[term] = sum([v ** 2 for v in wcp.values()])

    def dump(self, out_path):
        with open(out_path, 'w') as outfile:
            for k, v in self.gini.items():
                outfile.write("%s %f\n" % (k, v))

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the count data")
    cmd.add_argument("--output", help="path to dump the model", default=MODEL_PATH)
    args = cmd.parse_args()

    gini = GiniCoe(args.input)
    _logger.info("Training Gini coefficient from count file: %s" % args.input)
    gini.train()
    _logger.info("Dumping model to %s" % args.output)
    gini.dump(args.output)
