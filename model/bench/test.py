import cPickle as pickle
import os, sys, math
from train import NaiveBayes
UTIL = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(UTIL)
from util import *

TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../raw_data/aggregated/test.dat")

class NaiveDecoder(object):
    def __init__(self, model):
        self.model = model

    def decode(self, sentence):
        ret = dict()
        terms = sentence.split(' ')
        for domain in self.model.domain_backoff:
            ret[domain] = self.get_score(terms, domain)
        return ret

    def get_score(self, terms, domain):
        ret = 0.0
        for term in terms:
            if self.model.term_count[term] == 0:
                continue
            c = self.model.domain_backoff[domain] \
                if term not in self.model.domain_has[domain] \
                else self.model.count[term, domain]
            ret += math.log(float(c) / self.model.count[domain])
        return ret

def test(model):
    total = 0
    correct = 0
    decoder = NaiveDecoder(model)
    with open(TEST_FILE_PATH) as test_file:
        for line in test_file:
            line = line.strip()
            if not line:
                continue
            total += 1
            sentence, tag = line.split('\t')
            result = decoder.decode(sentence)
            predicted, _ = argmax(result.items())
            if predicted == tag:
                correct += 1
    print float(correct) / total

if __name__ == "__main__":
    model = pickle.load(open('naive.model'))
    test(model)
