#!/usr/bin/env python

import sys
import ROC_SVM
from scipy.sparse import *
from scipy import *
from sklearn import svm,linear_model
import pickle
from optparse import OptionParser,OptionValueError

def main():
    loss='squared_hinge'
    dual=True

    options = usage()

    ifile = options.ifile
    n_feats = options.n_feats
    C = float(options.C)
    do_rank = bool(options.do_rank)
    do_logistic_regression = bool(options.do_logistic_regression)
    n_feats = int(options.n_feats)
    model_of = options.model_of
    penalty=options.penalty
    
    if (penalty == 'l1'):
        dual=False
        loss='hinge'

    [features, labels] = ROC_SVM.read_features(ifile, n_feats)
    
    if (do_rank):
        [features,labels] = ROC_SVM.gen_ranking(features, labels)
    
    print "Done. Now feating"
    
    if (do_logistic_regression):
        dual=False
        clf = linear_model.LogisticRegression(C=C,penalty=penalty,dual=dual)
    else:
        clf = svm.LinearSVC(C=C,penalty=penalty,loss=loss,dual=dual)

    clf.fit(features, labels)
    pickle.dump(clf, open(model_of, 'wb'))
    # print clf.coef_[0][0]
    
    print "Done. Now testing"
    
    test = clf.predict(features)
    
    n_correct = 0
    
    ROC_SVM.evaluate(test,labels)



def usage():
    C=0.025
    penalty='l2'
    loss='squared_hinge'
    dual=True
    do_rank = 0
    do_logistic_regression = 0
   
    parser = OptionParser()
    give_tag=0
    n_training_samples = -1

    parser.add_option("-i", dest="ifile",
                    help="Input file", metavar="FILE")
    parser.add_option("-o", dest="model_of",
                    help="Model output file", metavar="FILE")
    parser.add_option("-n", dest="n_feats",
                    help="Number of features", metavar="INT")
    parser.add_option("-c", metavar="FLOAT",
                            dest="C", 
                            help="Regularization parameter",
                            default=C)
    parser.add_option("-p", metavar="STRING",
                            dest="penalty", 
                            help="Penalty (l1 or l2)",
                            default=penalty)
    parser.add_option("-r", dest="do_rank", default=False, action="store_true",
                            help="Use ranking")
    parser.add_option("-l", dest="do_logistic_regression", default=False, action="store_true",
                            help="Use logistic regression and not SVM")
                    
    
    (options, args) = parser.parse_args()

    if (options.ifile == None or options.model_of == None):
            raise OptionValueError("input file or model file missing")
    
    return options


    
sys.exit(main())