import sys, os
import numpy as np
from feat.terms.term_categorize import term_category
from sklearn.feature_extraction.text import TfidfVectorizer
from util.log import _logger

TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../data/aggregated/test.dat")
TRAIN_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../data/aggregated/train.dat")

def argmax(ls):
    if not ls:
        return None, 0.0
    return max(ls, key = lambda x: x[1])

def load_data(train_path):
    _logger.info("Loading data from %s" % train_path)
    X = []
    y = []
    with open(train_path) as train_file:
        for line in train_file:
            line = line.strip().decode('utf-8')
            if not line:
                continue
            terms, domain = line.split('\t')
            X.append(terms)
            y.append(domain)
    return np.array(X), np.array(y)

class Tokenizer(object):
    def __init__(self):
        pass

    def __call__(self, sentence):
        terms = sentence.strip().split(' ')
        ret = [term_category(term) for term in terms]
        return list(ret)

class Analyzer(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer(min_df = 1, binary = False, ngram_range = (1, 3),
                                     tokenizer = Tokenizer())
        self.tokens = self.tfidf.build_tokenizer()
        self.ngram = self.tfidf.build_analyzer()

    def __call__(self, sentence):
        ret = self.ngram(sentence)
        terms = self.tokens(sentence)
        for term in terms:
            cate = term_category(term)
            if term != cate:
                ret.append(cate)
        return ret



