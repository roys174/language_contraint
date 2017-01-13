#!/usr/bin/env python

import numpy as np
import random
import sys
import FileTools

def read_features(ifile, n_feats):
    print "Reading",ifile
    
    features = []
    labels = []
    with FileTools.openReadFile(ifile) as ifh:
        for line in ifh:
            d = line.rstrip().split("\t")
        
            if (len(d) < 3):
                print "Bad line:",line
                continue
        
            # print d,line
            labels.append(d[0])
        
            fs = d[2].split(" ")
        
            local_fs = [0 for i in range(n_feats)]
            for f in fs:
                [f,v] = f.split(":")
            
                local_fs[int(f)] = int(v)
        
#            local_fs_sparse = csr_matrix(local_fs)
#            print line,d[0],local_fs,local_fs_sparse.toarray()
#            break
            features.append(local_fs)
              
    ifh.close()
    
    return [features, labels]
    
def evaluate(test,labels,confidence = []):
    n_correct = 0
    for i in range(len(labels)):
        # print labels[i],test[i]
        n_correct += (labels[i]==test[i])
        # print labels[i],test[i],confidence[i]
    
    print n_correct,"/",len(labels),"=",n_correct*1./len(labels)

    if (len(confidence)):    
        n_pairs_correct = 0
    
        replace = {'2':'1', '1':'2'}
        for i in range(len(labels)/2):
            if (test[2*i] == test[2*i+1]):
                if (abs(confidence[2*i]) > abs(confidence[2*i+1])):
                    test[2*i+1] = replace[test[2*i+1]]
                else:
                    test[2*i] = replace[test[2*i]]
            
            # print labels[i],test[i]
            n_pairs_correct += (labels[2*i]==test[2*i])
            n_pairs_correct += (labels[2*i+1]==test[2*i+1])
    #        print labels[i],test[i],confidence[i]
    
        print n_pairs_correct,"/",len(labels),"=",n_pairs_correct*1./len(labels)


def gen_ranking(features, labels):
    features2 = []
    labels2 = []
    
    for i in range(len(features)/2):
        rand_bit = random.random()
        
        diff = np.array(features[2*i])-np.array(features[2*i+1])
        label = 1
        if (rand_bit > 0.5):
            diff = -diff
            label = 2

        features2.append(diff.tolist())
        labels2.append(label)

    print len(features2),len(labels2)
    return [features2, labels2]