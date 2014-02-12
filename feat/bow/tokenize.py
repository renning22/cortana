# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:10:34 2014

@author: Ning
"""

from util import *
from util.log import _logger
from feat.terms.term_categorize import term_category
import codecs

def parse(sentence):
    for term in sentence.split():
        yield term_category(term)
    
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