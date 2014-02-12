import numpy as np
import word2vec
import argparse
import codecs
from util import *

model = word2vec.load( conv.redirect('data|vectors.bin') )

nvocab = [ unicode(i,'utf-8') for i in model.vocab ]
index = { v:n for n,v in enumerate(nvocab) }
l2norm = model.l2norm

for i in index:
    print i


for filename in ['data|test.dat','data|train.dat']:
    filepath = conv.redirect(filename)
    
    out = codecs.open( filename.split('|')[1] + '.embed' , 'w', encoding='utf-8' )
    print 'out=%s' % (out)
    
    for row in tsv.reader(filepath):
    
        n = 0
        
        aggregate = np.zeros( model.l2norm.shape[1] )
        for term in row[0].split():
            ths = None
            if term in index:
                ths = l2norm[index[term]]
            else:
                print 'Not found %s' % (term)
            if ths is not None:
                aggregate += ths
                n += 1
        
        aggregate /= n
        aggre_str = ' , '.join( map(str,aggregate) )
        
        out.write( "%s\t%s\n" % (row[0],aggre_str) )
    