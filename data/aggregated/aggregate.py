# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:16:44 2014

@author: Ning

Extract lines from train and test data only regards to domain classification.
"""

import sys,os,re
import csv
from util import *


def aggregate():

    out_dic = dict()

    path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + "/../from_archive"

    for root, dirs, files in os.walk(path):
        for file in files:

            m = re.match(r"chs_(\w+)\.slot\.(\w+)\.tsv",file.lower())
            if m is not None:
                print "Train/Test: %s , %s" % m.groups()
                tag = m.group(2)
                if not out_dic.has_key(tag):
                    out_dic[tag] = open("%s.dat" % (tag),"w");
                with open(os.path.join(root,file),"r") as lines:
                    tsvin = csv.reader(lines,delimiter='\t')
                    for row in tsvin:
                        out_dic[tag].write( "%s\t%s\n" % (row[1],row[3]) )

            m = re.match(r"(\w+)\.chs\.snt",file.lower());
            if m is not None:
                print "Lexicon: %s" % m.groups()
                tag = m.group(1)
                if not out_dic.has_key(tag):
                    out_dic[tag] = open("%s.lexicon.dat" % (tag),"w")

                with open(os.path.join(root,file),"r") as lines:
                    for row in lines:
                        row = row.strip()
                        if row != '':
                            out_dic[tag].write( "%s\n" % (row) )

    for a, b in out_dic.items():
        b.close()
    
    print "Writing class lables"
    with open("class.dat",'w') as fl:
        clss = {i[1] for i in tsv.reader("train.dat")} | {i[1] for i in tsv.reader("test.dat")}
        fl.write( '\n'.join( sorted(list(clss)) ) )
    
if __name__ == "__main__":
    aggregate()