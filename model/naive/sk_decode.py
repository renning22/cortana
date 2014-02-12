import sys, os
import argparse
import cPickle as pickle
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import accuracy_score
from util.log import _logger
from util import *
from featurized.terms.term_categorize import term_category


TEST_FILE_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../raw_data/aggregated/test.dat")

def get_vert_idx(vert, term):
    return np.nonzero(vert.transform([term]))[1][0]

def get_nb_idx(vert, sel, term):
    return np.nonzero(sel.transform(vert.transform([term])))[1][0]

def serv(clf):
    vert = clf.named_steps['vert']
    sel = clf.named_steps['select']
    nb = clf.named_steps['nb']

    tokens = vert.build_analyzer()
    while True:
        query = raw_input('Input your query(must be segmented by SPACE), q to quit:\n').decode('utf-8')
        if query == u'q':
            break
        
        prob = clf.predict_log_proba([query])[0]
        domains = sorted(zip(nb.classes_.tolist(), prob, range(len(nb.classes_))), key = lambda x: -x[1])

        terms = [(term, sel.scores_[get_vert_idx(vert, term)]) for term in tokens(query)]
        terms = sorted(terms, key = lambda x: -x[1])
        for term, score in terms:
            print term, score
            
        print '\n%s\n' % ('=' * 30)
        
        for domain, val, _ in domains[:3]:
            print domain, val

        print '\n%s\n' % ('=' * 30)
        
        print 'TERM\t' + '\t'.join([domain for domain, _, _ in domains[:3]])
        for term, _ in terms:
            line = [term, ]
            for domain, _, di in domains[:3]:
                ti = get_nb_idx(vert, sel, term)
                val = nb.feature_log_prob_[di, ti]
                line.append(str(val))
            print '\t'.join(line)

        print '\n%s\n' % ('=' * 30)

    sys.exit(0)

def slim(sentence, clf):
    sel = clf.named_steps['select']
    vert = clf.named_steps['vert']
    terms = list(set(sentence.split()))
    terms = sorted([(term, sel.scores_[get_vert_idx(vert, term_category(term))]) for term in terms], 
                   key = lambda x: -x[1])[:7]
    return ' '.join([term[0] for term in terms])


def extract(X, clf):
    ret = []
    for sentence in X:
        ret.append(slim(sentence, clf))
    return ret
            

if __name__ == "__main__":
    _logger.info("loading model")
    clf = pickle.load(open('sk_naive.model'))
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    cmd.add_argument("--serv", help = "run as server", dest="as_server", action='store_true')
    args = cmd.parse_args()

    if args.as_server:
        serv(clf)

    X, y = load_data(args.path)

    # _logger.debug("Extracting merites for long sentences")
    # X = extract(X, clf)
    
    y_pred = clf.predict(X)
    outfile = open("predicted.dat", 'w')
    for i in range(len(y)):
        sentence, pred, gold = X[i], y_pred[i], y[i]
        outfile.write("%s\t%s\t%s\n" % (sentence.encode('utf-8'), pred.encode('utf-8'), gold.encode('utf-8')))

    _logger.info("accuracy: %f" % accuracy_score(y, y_pred))
    

