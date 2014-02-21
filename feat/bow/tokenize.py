# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:10:34 2014

@author: Ning
"""

from util import *
from util.log import _logger
from feat.terms.term_categorize import term_category
import codecs
import argparse

def parse(sentence):
    s1 = [term_category(term) for term in sentence.split()]
    if len(s1) == 0:
        return ["__empty__"]
    else:
        return s1

    # for i in xrange(len(s1)):
    #     if i > 0 and s1[i] == s1[i-1]:
    #         continue
    #     else:
    #         yield s1[i]
    
def tokenize():
    
    rows = tsv.reader(conv.redirect("data|train.dat"))
    with codecs.open("train.tokenized.dat",'w',encoding='utf-8') as fl:
        for row in rows:
            fl.write("%s\t%s\n" % (' '.join(list(parse(row[0]))) , row[1]) )
            
    rows = tsv.reader(conv.redirect("data|test.dat"))    
    with codecs.open("test.tokenized.dat",'w',encoding='utf-8') as fl:
        for row in rows:
            fl.write("%s\t%s\n" % (' '.join(list(parse(row[0]))) , row[1]) )
    
if __name__ == "__main__":
    
    tokenize()
