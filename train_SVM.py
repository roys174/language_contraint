#!/usr/bin/env python

import sys
sys.path.insert(0, '/Users/roysch/aristo/entailment/scripts/')
import ROC_SVM
from scipy.sparse import *
from scipy import *
from sklearn import svm
import pickle

def main():
    C=0.025
    do_rank = 0
    argc=len(sys.argv)
    if (argc < 4):
        print "Usage:",sys.argv[0],"<if> <n features> <of> <C="+str(C)+"> <do rank>"
        return -1
    elif (argc > 4):
        C=float(sys.argv[4])
        if (argc > 5):
            do_rank = bool(sys.argv[5])
            
    n_feats = int(sys.argv[2])

    [features, labels] = ROC_SVM.read_features(sys.argv[1], n_feats)
    
    if (do_rank):
        [features,labels] = ROC_SVM.gen_ranking(features, labels)
    
    print "Done. Now feating"
    
    clf = svm.LinearSVC(C=C)
    clf.fit(features, labels)
    pickle.dump(clf, open(sys.argv[3], 'wb'))
    
    print "Done. Now testing"
    
    test = clf.predict(features)
    
    n_correct = 0
    
    ROC_SVM.evaluate(test,labels)


    
sys.exit(main())