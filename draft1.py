
# coding: utf-8

# ## Authors
# - Alexander USOLTSEV
# - Nicolas GRANGER

# # Setup notebook

# # Load and preprocess dataset

from data import *
from preprocessor import *
from sklearn import cross_validation
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from bs4 import BeautifulSoup
import random

setup('/home/granger/dev/kaggledato/data')

labels = readLabels()

# Balance classes
labels = readLabels()
pos = [k for k, v in labels.items() if v]
neg = [k for k, v in labels.items() if not v]
random.shuffle(pos)
random.shuffle(neg)
balanced_labels = {k: True for k in pos[:1500]}
balanced_labels.update({k: False for k in neg[:1500]})

keys = list(balanced_labels.keys())

trainId, testId = cross_validation.train_test_split(np.array(keys[:3000]), test_size=0.2, random_state=10)
trainY = [labels[id] for id in trainId]
testY  = [labels[id] for id in testId]

# Extract words statistics
vectorizer = TfidfVectorizer(sublinear_tf = True, 
                             max_df       = 0.5,
                             #min_df       = 0.005,
                             preprocessor = lambda s : html2txt(purge_html(BeautifulSoup(s, 'html.parser'))),
                             #tokenizer    = nltk.tokenize.WhitespaceTokenizer,
                             stop_words='english')
trainWStats = vectorizer.fit_transform(readSamples(trainId))
testWStats  = vectorizer.transform(readSamples(testId))

# reduce dimension
selector = SelectPercentile(f_classif, percentile=1)
selector.fit(trainWStats, trainY)
trainX = selector.transform(trainWStats)
testX  = selector.transform(testWStats)


# # Build and train Model

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import mixture

"""
clf = GaussianNB()
clf = mixture.GMM(n_components=100)
"""
clf = svm.SVC(C = 2.0,
              gamma=2,
              class_weight = 'auto',
              probability=True,
              verbose = True,
              tol=1e-5)
clf.fit(trainX.toarray(), trainY)
predictions = clf.predict(trainX.toarray())
pred_acc_score = accuracy_score(trainY, predictions)
print('Training score: ' + str(pred_acc_score))


# # Test

predictions = clf.predict(testX.toarray())
pred_acc_score = accuracy_score(testY, predictions)
print('Testing score: ' + str(pred_acc_score))
