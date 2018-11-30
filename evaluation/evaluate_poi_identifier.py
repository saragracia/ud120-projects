#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
allzeros = [0] * len(pred)

print "How many poi's", sum(pred)
print "How many people in test set", len(pred)

print accuracy_score(pred, labels_test)
print "accuracy all non poi's", accuracy_score(allzeros, labels_test)
truepositives = [a and b for a,b in zip(pred, labels_test)]
print "True positives", truepositives

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print "precision", precision_score(labels_test, pred)
print "recall", recall_score(labels_test, pred)

print "------------------------------------------------------------------"

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print accuracy_score(predictions, true_labels)
truepositives = [a and b for a,b in zip(predictions, true_labels)]
print "True positives", truepositives, sum(truepositives)

truenegatives = [ not(a or b) for a,b in zip(predictions, true_labels)]
print "True negatives", truenegatives, sum(truenegatives)

print "precision", precision_score(true_labels, predictions)
print "recall", recall_score(true_labels, predictions)
print "f1_score", f1_score(true_labels, predictions)

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(true_labels, predictions)
print CM
print "True negatives", CM[0][0]
print "FAlse negatives", CM[1][0]
print "True positives", CM[1][1]
print "False positives", CM[0][1]
print "------------------------------------------------------------------"
