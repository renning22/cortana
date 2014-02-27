import numpy as np
from scipy.sparse import *
import word2vec
import argparse
import codecs
import sys
import cPickle as pickle
from util import *


def load(modelpath):

    model = word2vec.load(modelpath)

    nvocab = [ unicode(i,'utf-8') for i in model.vocab ]
    index = { v:n for n,v in enumerate(nvocab) }
    l2norm = model.l2norm
    
    return (index,l2norm)

def vectorize(modelpath,binary=True):

    index , l2norm = load(modelpath)
    
    for filename in ['data|test.dat','data|train.dat']:
    
        if not binary:
            out = codecs.open( filename.split('|')[1].split('.')[0] + '.vectorized.dat' , 'w', encoding='utf-8' )
            print 'out=%s' % (out)

        mat = []
        filepath = conv.redirect(filename)
        for i,row in enumerate(tsv.reader(filepath)):
        
            if i%1000 == 0:
                sys.stdout.write("%s\r" % i)
                sys.stdout.flush()

            n = 0
            aggregate_vec = np.zeros( l2norm.shape[1] )
            for term in row[0].split():
                for cha in term:
                    ths = None
                    if cha in index:
                        ths = l2norm[index[cha]]
                    else:
                        pass
                        #print 'Not found %s' % (term)
                    if ths is not None:
                        aggregate_vec += ths
                        n += 1
            
            if n > 0:
                aggregate_vec /= n
             
            mat.append( aggregate_vec )
            
            if not binary:
                aggregate_vec_str = ' , '.join( map(str,aggregate_vec) )
                out.write( "%s\t%s\n" % (row[0],aggregate_vec_str) )

        if binary:
            print "Dumping binary: " + filename.split('|')[1].split('.')[0] + '.vectorized.mat'
            npmat = np.matrix( mat , dtype=np.float32 )
            pickle.dump(npmat,open( filename.split('|')[1].split('.')[0] + '.vectorized.mat' ,'w'))

        
if __name__ == "__main__":
    vectorize( conv.redirect('data|newsarticles.aggregated.zh-cn.snapshotTitle.preprocessed.shufed.w2v.bin') )
        
