# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 11:19:56 2018

@author: Surya
"""
import csv
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

speech = raw_input("Enter a sentense\n")


train = []
test = []

with open("training.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        train.append(row)
        
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        test.append(row)


cl = NaiveBayesClassifier(train)
cl.classify("This is an amazing library!")
prob_dist = cl.prob_classify("This one's a doozy.")
prob_dist.max()
round(prob_dist.prob("machine"), 2)
round(prob_dist.prob("no machine"), 2)
blob = TextBlob(speech, classifier=cl)
blob.classify()
for s in blob.sentences:
    print("\n\n\n" + str(s))
    print("\n" + str(s.classify()))
    #print("\n" + str(cl.accuracy(test)))
    