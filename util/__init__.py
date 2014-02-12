from featurized.terms.term_categorize import term_category
from util.log import _logger

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

class Analyzer(object):
    def __init__(self):
        pass

    def __call__(self, sentence):
        terms = sentence.strip().split(' ')
        ret = [term_category(term) for term in terms]
        return list(ret)
