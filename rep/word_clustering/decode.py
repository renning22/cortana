import os, sys
CWD = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

g_word_class = dict()

def load(path):
    with open(path) as class_file:
        for line in class_file:
            line = line.strip()
            if line:
                word, cluster = line.split('\t')
                word = word.decode('utf-8')
                g_word_class[word] = cluster.decode('utf-8')

def get_cluster(word):
    return g_word_class[word]

load(CWD + "/clusters.txt")
