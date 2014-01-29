import cPickle as pickle
import os, sys, math
from train import NaiveBayes
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util import *
from featurized.terms.term_categorize import term_category

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
        ret = math.log(float(self.model.count[domain]) / self.model.training_sentence_count)
        for term in terms:
            term = term_category(term)
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
            # else:
            #     print sentence, predicted, tag
    print float(correct) / total

def serv(model):
    decoder = NaiveDecoder(model)
    while True:
        query = raw_input('Input your query(must be segmented by SPACE), q to quit:\n')
        if query == 'q':
            return
        ret = decoder.decode(query)
        for domain, score in sorted(ret.items(), key = lambda x: -x[1]):
            print domain, score
        

if __name__ == "__main__":
    model = pickle.load(open('naive.model'))
    if len(sys.argv) > 1 and sys.argv[1] == '--serv':
        serv(model)
    else:
        test(model)
