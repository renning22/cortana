import sys, os
# what about data aggregate importing this
#from feat.terms.term_categorize import term_category
from util.log import _logger

root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../")

TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../data/aggregated/test.dat")
TRAIN_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../data/aggregated/train.dat")

def argmax(ls):
    if not ls:
        return None, 0.0
    return max(ls, key = lambda x: x[1])

def load_data(train_path):
    _logger.info("Loading training data from %s" % train_path)
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
    return X, y

class Tokenizer(object):
    def __init__(self):
        pass

    def __call__(self, sentence):
        terms = sentence.strip().split(' ')
        ret = [term_category(term) for term in terms]
        return list(ret)

__all__ = ["tsv", "conv", "log", "Tokenizer", "load_data", "argmax", "TEST_FILE_PATH", "TRAIN_FILE_PATH"]


