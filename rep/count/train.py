# Count the co-occurance of each (term, domain) pair

import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util.log import _logger
from feat.terms.term_categorize import term_category

DEFAULT_OUTPATH = "./count.dat"

class Counter(object):
    def __init__(self, train_path):
        self.train_path = train_path

    def train(self):
        self.count = defaultdict(int)
        c = 0
        with open(self.train_path) as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                terms, domain = line.split('\t')
                term_set = set()
                for term in terms.split(' '):
                    term = term_category(term)
                    if term not in term_set:
                        term_set.add(term)
                        self.count[(term, domain)] += 1
                c += 1
                if c % 10000 == 0:
                    _logger.debug("%d records processed" % c)

    def dump(self, path):
        with open(path, 'w') as outfile:
            for key, val in self.count.items():
                term, domain = key
                outfile.write("%s %s %d\n" % (term.encode('utf-8'), domain.encode('utf-8'), val))
                


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data")
    cmd.add_argument("--output", help="path to dump the model", default=DEFAULT_OUTPATH)
    args = cmd.parse_args()

    counter = Counter(args.input)
    _logger.info("training from %s" % args.input)
    counter.train()
    _logger.info("dumping model to %s" % args.output)
    counter.dump(args.output)
