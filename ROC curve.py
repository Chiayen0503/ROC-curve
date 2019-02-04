#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:14:12 2019

@author: ChiaYen
"""

# credit by:
# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
import pandas as pd
from matplotlib import pyplot as plt


plt.style.use('ggplot')


df = pd.read_csv('breast-cancer.data.txt',sep=",")
df.columns = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
df.head() #check first few row data
df["deg-malig"] = df["deg-malig"].astype(object)

df.shape #check shape: (285, 10)
pd.isnull(df).values.any() #check missing value: False
df['class'].value_counts() #check if the data is balanced 

[df[x].unique().shape[0] for x in df.columns] 

Y = df['class']
X = df[df.columns[1:]]


X_dummy = pd.get_dummies(X)
Y_dummy = Y.apply(lambda x: 0 if x=='no-recurrence-events' else 1) # 1 is recurrence-events, i.e, target


from sklearn import tree
from sklearn import model_selection

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# calculate the fpr and tpr for all thresholds of the classification
for i in range(6):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_dummy, Y_dummy, test_size=0.2)
    clf = tree.DecisionTreeClassifier(min_samples_leaf=1, criterion ='entropy', splitter ='best',max_depth = 80)
    clf = clf.fit(X_train, Y_train)
       
    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()