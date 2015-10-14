# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:56:49 2015

@author: Alexander USOLTSEV
"""
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import os
from bs4 import BeautifulSoup
import re
import nltk
#nltk.download()  # Download text data sets, including stop words. Do it only ones!
from nltk.corpus import stopwords # Import the stop word list

#%% Helpers here
def timer(f):
    """ Decorator to estimate how much time a prodived function f takes to run.
    """
    def _f(*args):
        from time import time
        start_time = time()
        result = f(*args)
        end_time = time()
        print "Function %s took %f s to execute" % (f.__name__, \
                                                (end_time-start_time))
        return result                                       
    return _f


#%% Let's import train.csv with all ground truth labels
train_ground_truth = u'./raw-data/train.csv'
with open(train_ground_truth, 'r') as f:
    train_df = pd.read_csv(f)
    
# Let's show some basic info about train.csv
print train_df.info()
print train_df.describe()

#%% Let's read html files to a list
raw_html_files_path_0 = u'./raw-data/0/'
raw_html_files_0 = os.listdir(raw_html_files_path_0)

#%% Now let's create dataframe with ground truth labels only for existing html files
train_0 = train_df.file.isin(raw_html_files_0)
train_0 = train_df[train_0]
print train_0.info()
print "Organic texts: %d, sponsored texts: %d" % (sum(train_0.sponsored==0), sum(train_0.sponsored==1))

#%%
def html_to_text(raw_html):
    """ Function to convert a raw html to a clean string of words
        The input is a single string (a raw html code), and 
        the output is a single string (a preprocessed blog text)
    """
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_html).get_text() 
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  

#%%
def text_from_html_file(html_dir, html_filename):
    """ Function to read raw html files one by one and extract clean text from
        html. The input is a list of filenames and full path to the 
        containing dir. The output is a list of strings.
    """
    # 1. Join filename and path
    html_filename = os.path.join(html_dir,html_filename)
    
    # 2. Read html and return clean text
    with open(html_filename, 'r') as f:
        raw_html = f.read()
    return html_to_text(raw_html)
    
#%% Append a column to the train dataframe with a text extracted from html code 
# ATTENTION! It will take 30-40 minutes for part 0
train_0['text'] = [text_from_html_file(raw_html_files_path_0,raw_html) for raw_html in train_0.file]

#%% Now it's time to train basic model! We will split train data to have some test 
# to play with. Also we'll call article's text as features and flag 'sponsored' 
# as labels.  
# 1. Let's split train data to subsets: train and test
part_0_train, part_0_test = cross_validation.train_test_split(train_0, test_size=0.2, random_state=10)
features_train = np.array(part_0_train.text)
labels_train = np.array(part_0_train.sponsored)
features_test = np.array(part_0_test.text)
labels_test = np.array(part_0_test.sponsored)

#%% To do machine learning we should use not words, but vector representation.

# We are using Word Vectorizer (Bag of Words) with TfIdf (term frequency–inverse
# document frequency) normalization.
def feature_transform(features_train, features_test, top_percent=1):
    """ Function to apply Bag of Words feature creator with TfIdf statistic 
        normalisation. The input is train and test text, and optional parameter
        'top_percent' which shows how many percent of super high dimensional
        text feature space is to return (defaul is 1%). 
        The output is the transformed train and test feature vectors suitable 
        to use with sklearn classifiers.
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)
    
    ### Feature selection, because text is super high dimensional
    selector = SelectPercentile(f_classif, percentile=top_percent)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()
    return features_train_transformed, features_test_transformed
    
#%%
# Convert text features to vector features
features_train, features_test = feature_transform(features_train, features_test)

#%% Train time!
@timer
def naive_bayes(features_train, features_test, labels_train, labels_test):
    clf = GaussianNB()
    fit = timer(clf.fit)
    fit(features_train, labels_train)
    
    predict = timer(clf.predict)
    labels_predicted = predict(features_test)
    clf_report = metrics.classification_report(labels_test, labels_predicted)
    
    return labels_predicted, clf_report
#%%
# Use Naive Bayes to predict where article is sponsored or not    
_, clf_report = naive_bayes(features_train, features_test, labels_train, labels_test)
print "Classification report: \n%s" % clf_report