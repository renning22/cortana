# coding=utf-8

import os,sys
import codecs
import re
import math

input = sys.argv[1];
output = input.split('.')
output[-1:] = ["preprocessed"] + output[-1:]
output = '.'.join(output)
print "Output to: " + output

ofs = codecs.open(output,'w','utf-8')
for i,rawline in enumerate(codecs.open(input, 'r', 'utf-8')):

    if i%1000 == 0:
        sys.stdout.write('%d\r' % i)
        sys.stdout.flush()

    rawline = rawline.split('\t')

    line = rawline[0].strip()

    nline = []
    for term in line.split():
        if term is '':
            continue
    
        if re.match(ur'[\\/+\-_]+',term):
            continue

        if re.match(r'[0-9]+', term):
            nline.append( u"__DIGITS__" )
        elif re.match(r'^[0-9]{1,2}[ap]m$', term):
            nline.append( u"__TIMEAMPM__" )
        elif re.match(r"[\w0-9]{1,2}",term):
            nline.append( u"__ALPHABET__" )
        else:
            nline.append( term )


    nline.append( u"__STOP__" )

    rawline[0] = ' '.join(nline)
    ofs.write( "%s" % ('\t'.join(rawline)) )
