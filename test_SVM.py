#!/usr/bin/env python

import sys
sys.path.insert(0, '/Users/roysch/aristo/entailment/scripts/')
import ROC_SVM
import FileTools
from scipy.sparse import *
from scipy import *
from sklearn import svm
import pickle


def main():
    do_rank = 0
    ids=None
    gold=None
    out_file = None
    if (len(sys.argv) < 3):
        print "Usage:",sys.argv[0],"<if> <model if> <is ranking> <mapping file> <out file>" 
        return -1
    elif (len(sys.argv) > 3):
        do_rank = int(sys.argv[3])
        if (len(sys.argv) > 4):
            mapping_file = sys.argv[4]
            ids,gold = read_mapping_file(mapping_file)
            if (len(sys.argv) > 5):
                out_file = sys.argv[5]

    clf = pickle.load(open(sys.argv[2], 'r'))
    n_feats = len(clf.coef_[0])
    
    features = []
    labels = []

    [features, labels] = ROC_SVM.read_features(sys.argv[1], n_feats)
    
    if (do_rank):
        [features,labels] = ROC_SVM.gen_ranking(features, labels)
    
    print "Done. Now testing"
    
    test = clf.predict(features)
    confidence = []
    
    if (not do_rank):
        confidence = clf.decision_function(features)
    
    
    ROC_SVM.evaluate(test,labels,confidence,ids,gold, out_file)
    

def read_mapping_file(ifile):
    ids = []
    gold = []
    with FileTools.openReadFile(ifile) as ifh:
        # Ignore first lione
        ifh.readline()

        for line in ifh:
            e = line.rstrip().split("\t");
            id = e[0]
            ids.append(id)
            gold.append(int(e[-1]))
	
        ifh.close()

	return ids,gold


sys.exit(main())