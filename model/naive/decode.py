import os, sys, math
import cPickle as pickle
import argparse
import numpy as np

from train import NaiveBayes
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(ROOT)
from util import *
from util.log import _logger
from featurized.terms.term_categorize import term_category, g_term_count

TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../raw_data/aggregated/test.dat")

class NaiveDecoder(object):
    def __init__(self, model):
        self.model = model
        self.backoff = np.mean(self.model.domain_backoff.values())

    def decode(self, sentence):
        ret = dict()
        terms = sentence.split(' ')
        for domain in self.model.domains:
            ret[domain], _ = self.get_score(terms, domain)
        return ret

    def predict(self, sentence):
        ret = self.decode(sentence)
        return max(ret.items(), key = lambda x: x[1])[0]

    def term_score(self, term, domain):
        assert(type(term) == type(domain) == unicode)
        #backoff = self.model.domain_backoff[domain]
        backoff = self.backoff
        c = backoff if term not in self.model.domain_has[domain] \
            else self.model.count[term, domain]
        val = math.log(float(c) / self.model.count[domain], 10.0)
        assert val < 0
        return -math.pow(-val, 1.0 / 10.0)

    def get_score(self, terms, domain):
        # a priori of domain distribution here doesn't make much sense, the value should be from live data
        val = 0.0
        detail = {'__priori__': val}
        term_set = set()
        for term in terms:
            term = term_category(term)
            if term in term_set:
                continue
            term_set.add(term)
            score = self.term_score(term, domain)
            detail[term] = score
            val += score
        return val, detail

def test(model, test_file_path = TEST_FILE_PATH):
    total = 0
    correct = 0
    decoder = NaiveDecoder(model)
    outfile = open("predicted.dat", 'w')
    _logger.info("Testing %s" % test_file_path)
    with open(test_file_path) as test_file:
        processed = 1
        for line in test_file:
            line = line.strip().decode('utf-8')
            if not line:
                continue
            total += 1
            sentence, tag = line.split('\t')
            result = decoder.decode(sentence)
            predicted, _ = argmax(result.items())
            outfile.write("%s\t%s\t%s\n" % (sentence.encode('utf-8'), predicted.encode('utf-8'), tag.encode('utf-8')))
            if predicted == tag:
                correct += 1
            if processed % 1000 == 0:
                _logger.debug("%d lines processed" % processed)
            processed += 1
    outfile.close()
    _logger.info("accuracy: %f" % (float(correct) / total))

def serv(model):
    decoder = NaiveDecoder(model)
    while True:
        query = raw_input('Input your query(must be segmented by SPACE), q to quit:\n').decode('utf-8')
        if query == u'q':
            return
        domains = raw_input('Input the domains you want to compare:\n').decode('utf-8')
        if not domains:
            domains = decoder.model.domains
        else:
            domains = domains.split(' ')
        ret = decoder.predict(query)
        print "\n%s\n%s\n" % (ret, '=' * 50)
        
        lst = []
        for domain in domains:
            score, detail = decoder.get_score(query.split(' '), domain)
            lst.append((score, domain, detail))
        lst.sort(key = lambda x: -x[0])
        for domain, score, detail in lst:
            print score, domain
            for term in query.split(' '):
                cate = term_category(term)
                sys.stdout.write('%s(%s, %d): %.4f\t' % (term, cate, g_term_count[term], detail[cate]))
            print '\n%s\n' % ('-' * 20)        

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--serv", help = "run as server", default=False, dest="as_server", action='store_true')
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    args = cmd.parse_args()

    _logger.info("Loading model")
    model = pickle.load(open('naive.model'))


    if args.as_server:
        serv(model)
    else:
        test(model, args.path)