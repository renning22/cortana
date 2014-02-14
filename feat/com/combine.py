import numpy as np
from scipy.sparse import hstack
import cPickle as pickle
from util import *



w2v_train = pickle.load(open(conv.redirect('w2v|train.vectorized.mat')))
w2v_test = pickle.load(open(conv.redirect('w2v|test.vectorized.mat')))

bow_train = pickle.load(open(conv.redirect('bow|train.vectorized.mat')))
bow_test = pickle.load(open(conv.redirect('bow|test.vectorized.mat')))


com_train = hstack( [bow_train,w2v_train] , dtype=np.float32 )
com_test = hstack( [bow_test,w2v_test] , dtype=np.float32 )

pickle.dump( com_train, open('train.vectorized.mat','w') )
pickle.dump( com_test, open('test.vectorized.mat','w') )

