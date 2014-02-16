# -*- coding: utf-8 -*-
"""
Created on Sun Feb 02 11:06:56 2014

For convention or convenience

short name for path

E.g.
    data|train.dat = \data\aggregated\train.dat
    naive.model = \model\naive\naive.model
    terms.dat = \feat\terms
    naive|predicted.dat = \model\naive\predicted.dat
    
@author: Ning
"""

import os,sys
from log import _logger

root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../")

def argmax(ls):
    if not ls:
        return None, 0.0
    return max(ls, key = lambda x: x[1])

def expand_list(ls):
    ret = []
    for i in ls:
        ret.extend(i)
    return ret
    
def subsequence(a,b):
    base = int(0)
    for i in b:
        try:
            base = a.index(i,base)+1
        except:
            return False
    return True
    
def redirect(inpt):
    candidates = []
    terms = inpt.strip().lower().decode('utf-8').split('|')
    terms = expand_list([i.split('_') for i in terms])
    terms = expand_list([i.split('.') for i in terms])
    for dir, dirs, files in os.walk(root):
        for file in files:
            rel_path = os.path.normpath( os.path.normcase(os.path.relpath(dir+"/"+file,root)) );
            
            #print "%s,%s,%s , %s" % (dir,dirs,file,rel_path)

            path_terms = rel_path.lower().split('\\')
            path_terms = expand_list([i.split('/') for i in path_terms])
            path_terms = expand_list([i.split('.') for i in path_terms])
            path_terms = expand_list([i.split('_') for i in path_terms])

            #print terms
            #print path_terms
            
            if subsequence(path_terms,terms):
                candidates.append(rel_path)
                
    if len(candidates) == 0:
        _logger.warning('Not found "%s"' % (inpt))
        return ""
    elif len(candidates) >= 1:
        if len(candidates) > 1:
            _logger.warning('Ambiguious for "%s":' % (inpt))
            for i in candidates:
                _logger.warning("\t%s\n"%(i))

        ret = os.path.normcase( os.path.abspath(root+"/"+candidates[0]) )
        _logger.info('Redirect "%s" to "%s"' % (inpt,ret))
        return ret

if __name__ == "__main__":
    redirect("naive|predicted.dat")
