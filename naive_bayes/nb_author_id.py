#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)  
print "training time:", round(time()-t0, 3), "s"    #0.899

t1 = time()
preds = clf.predict(features_test)
print "predict time 1:", round(time()-t1, 3), "s"   #0.123
print accuracy_score(preds, labels_test)

t2 = time()
accuracy = clf.score(features_test, labels_test)
print "predict time 2:", round(time()-t2, 3), "s"   #0.122
print accuracy 

#########################################################
