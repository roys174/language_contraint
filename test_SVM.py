#!/usr/bin/env python

import sys
sys.path.insert(0, '/Users/roysch/aristo/entailment/scripts/')
import ROC_SVM
from scipy.sparse import *
from scipy import *
from sklearn import svm
import pickle


def main():
    do_rank = 0
    if (len(sys.argv) < 4):
        print "Usage:",sys.argv[0],"<if> <n_feats> <model if> <is ranking>" 
        return -1
    elif (len(sys.argv) > 4):
        do_rank = bool(sys.argv[4])

    n_feats = int(sys.argv[2])
    clf = pickle.load(open(sys.argv[3], 'r'))
    
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
    
    
    ROC_SVM.evaluate(test,labels,confidence)
    

sys.exit(main())