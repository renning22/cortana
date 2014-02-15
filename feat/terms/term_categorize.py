# coding=utf-8
# This script assumes there's term freqency count file(terms.dat) in same directory.
import sys, os, re
from glob import glob
from collections import defaultdict

PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + "/terms.dat"
RARE = 5
LEXICON = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + "/../../data/aggregated/*.lexicon.dat"

g_term_count = defaultdict(int)
g_lexicon = defaultdict(set)

def load():
    with open(PATH) as count_file:
        for line in count_file:
            line = line.strip()
            if line:
                line = line.split()
                term = line[0].decode('utf-8')
                freq = int(line[1])
                g_term_count[term] = freq

    for lexicon_file in glob(LEXICON):
        tag = u'__%s__' % os.path.basename(lexicon_file).split('.')[0].upper()
        with open(lexicon_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    g_lexicon[tag].add(line.decode('utf-8'))

def term_category(term):
    if type(term) != unicode:
        term = term.decode('utf-8')

    DIGITS = r'^[0-9]+$'
    CH_DIGITS = ur'^[零一二三四五六七八九十百千万]+$'
    for tag in g_lexicon:
        if term in g_lexicon[tag]:
            return tag
    if re.match(DIGITS, term):
        return u"__DIGITS__"
    if re.match(CH_DIGITS, term):
        return u"__CHDIGITS__"
    if re.match(r'^[0-9]{1,2}[ap]m$', term):
        return u"__TIMEAMPM__"
    if g_term_count.setdefault(term, 0) <= RARE:
        return u"__RARE__"
    return term

def gen_categorized_terms(path):
    found = set()
    outfile = open(path, 'w')
    for term in g_term_count:
        term = term_category(term)
        if term not in found:
            found.add(term)
            outfile.write("%s\n" % term.encode('utf-8'))
    outfile.close()

g_categorized_term_idx = dict()
def load_categorized_terms(path):
    with open(path) as infile:
        for line in infile:
            term = line.strip().decode('utf-8')
            g_categorized_term_idx[term] = len(g_categorized_term_idx)

def get_categorized_term_idx(term):
    return g_categorized_term_idx[term]

load()

