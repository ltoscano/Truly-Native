# Setup environment
import os
dataDir = '/run/media/nicolas/My_Passport/data/kaggle-dato'
os.environ['NLTK_DATA'] = os.path.join(dataDir, 'nltk_data')

# Imports

import random
import numpy as np

import nltk
from bs4 import BeautifulSoup

from sklearn import cross_validation
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# project functions

from data import setup, read_labels, read_samples
from preprocessor import purge_HTML, html2txt, count_links, avg_sentence_len

# Setup

setup(dataDir)

nTrain = 700
nTest  = 500

# Build dataset
labels = read_labels()
pos = [k for k, v in labels.items() if v]
neg = [k for k, v in labels.items() if not v]
random.shuffle(pos)
random.shuffle(neg)
balanced_labels = {k: True for k in pos[:nTrain]}
balanced_labels.update({k: False for k in neg[:nTest]})

trainId, testId = cross_validation.train_test_split(
    np.array(list(balanced_labels.keys())), 
    train_size=nTrain, 
    test_size=nTest, 
    random_state=10)
trainRaw  = read_samples(trainId)
testRaw   = read_samples(testId)
trainY    = [labels[id] for id in trainId]
testY     = [labels[id] for id in testId]

# Extract text statistics and text
trainStats = np.empty([nTrain, 3])
testStats  = np.empty([nTest, 3])
# tricky: text loader and parser (for word stat extraction) also saves
# text statistics on the fly, hence this loader wrapper
def featExtractor(sourceIter, stats):
    i = 0
    for html in sourceIter:
        soup   = purge_HTML(BeautifulSoup(html, 'html.parser'))
        (intern, extern) = count_links(soup)
        text   = html2txt(soup)
        avgLen = avg_sentence_len(text)
        stats[i][0] = intern
        stats[i][1] = extern
        stats[i][2] = avgLen
        yield text
        i += 1
        if i >= stats.shape[0]:
            break
        
                        
# Extract features
vectorizer = TfidfVectorizer(sublinear_tf = True, 
                             max_df       = 0.5,
                             #min_df       = 0.005,
                             #tokenizer    = nltk.tokenize.WhitespaceTokenizer,
                             stop_words=nltk.corpus.stopwords.words('english'))
trainWStats = vectorizer.fit_transform(featExtractor(trainRaw, trainStats))
testWStats  = vectorizer.transform(featExtractor(testRaw, testStats))

# Reduce dimension
selector = SelectPercentile(f_classif, percentile=1)
selector.fit(trainWStats, trainY)
trainX = selector.transform(trainWStats)
testX  = selector.transform(testWStats)

# Merge data types
trainX = np.concatenate((trainX.toarray(), trainStats), axis=1)
testX  = np.concatenate((testX.toarray(), testStats), axis=1)

# Build and train Model
#clf = svm.SVC(C = 2.0,
#              gamma=2,
#              class_weight = 'auto',
#              probability=True,
#              verbose = True,
#              tol=1e-5)
clf = LogisticRegression(C=1e2, class_weight = 'auto')
clf.fit(trainX, trainY)
predictions = clf.predict(trainX)
pred_acc_score = accuracy_score(trainY, predictions)
print('Training score: ' + str(pred_acc_score))
print(confusion_matrix(predictions, trainY))

# # Test

predictions = clf.predict(testX)
pred_acc_score = accuracy_score(testY, predictions)
print('Testing score: ' + str(pred_acc_score))
print(confusion_matrix(predictions, testY))
