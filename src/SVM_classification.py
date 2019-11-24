# coding=utf-8
import numpy
import sys
from time import time
from sklearn.metrics import accuracy_score
from sklearn import svm
import warnings
sys.path.append("../tools/")
from pre_process_data import pre_proc
warnings.filterwarnings('ignore')

low_pred=0
features_train, features_test, labels_train, labels_test=pre_proc()

if(len(features_test)==0):
    sys.exit("No test features found")

print("Training set size: " +str(len(features_train)))
print("Test set size: " +str(len(features_test)))

clf = svm.SVC(kernel='rbf',C=10000, gamma='auto', probability=True)
t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
predicted_labels=clf.predict(features_test)
probability_labels=clf.predict_proba(features_test)
print "predicting time:", round(time()-t1, 3), "s"
prob_ordered= numpy.argsort(probability_labels, axis=1)[:, -2:]
for predicted_val, prob_ind, indices in zip(predicted_labels, probability_labels, prob_ordered):
    prob=round(prob_ind[indices[1]]*100, 3)
    if(prob<15):
        low_pred+=1

if(len(features_test)>0):
    low_pred_conf=round(low_pred*100/len(features_test),2)

print("Accuracy: "+str(round(accuracy_score(labels_test, predicted_labels)*100,2))+"%")
print(str(low_pred_conf)+"% predicted with confidence less than 15")