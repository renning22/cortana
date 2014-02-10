import sys, os

GINI_MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/gini.dat"


gini = dict()
sorted_terms = list()
term_num = 0

with open(GINI_MODEL_PATH) as infile:
    for line in infile:
        term, val = line.split()
        term = term.decode('utf-8')
        val = float(val)
        gini[term] = val

term_num = len(gini)
sorted_terms = sorted(gini.items(), key = lambda x: -x[1])

def top(k):
    return [term for term, val in sorted_terms[:k]]

def get_gini(term):
    return gini[term] if term in gini else 0

        
