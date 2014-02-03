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

    DIGITS = r'[0-9]+$'
    CH_DIGITS = ur'[零一二三四五六七八九十千百万亿]+$'
    for tag in g_lexicon:
        if term in g_lexicon[tag]:
            return tag
    if re.match(DIGITS, term):
        return u"__DIGITS_"
    if re.match(CH_DIGITS, term):
        return u"__CHDIGITS__"
    if g_term_count.setdefault(term, 0) <= RARE:
        return u"__RARE__"
    return term

load()

