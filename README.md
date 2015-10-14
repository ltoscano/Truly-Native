# Truly Native: Baseline Classifier
Python code to implement the baseline classification system for the Kaggle competition "Truly Native". In two words, in this competition you need to predict whenever an internet article is sponsored (aka paid) or not. More details on [Truly Native](https://www.kaggle.com/c/dato-native) competition's page.

## Technical Notes
* To run python code you should download the [training data](https://www.kaggle.com/c/dato-native/data) in advance.

## Requirements

* Tested with Python 2.7
* pandas
* numpy
* sklearn
* nltk
* beautifulsoup

## Basic Idea
This baseline classification system uses Part 0 of training data for "Truly Native" competition to learn TfIdf Vector Feature for both sponsored and not sponsored internet articles.

After that these feature vectors are used to train Naive Bayes classifier and perform prediction.

Prediction results represented as a classification report.

## Sample output
**Classification report:**

             precision    recall  f1-score   support

          0       0.94      0.94      0.94      3653
          1       0.43      0.43      0.43       395
        avg       0.89      0.89      0.89      4048

*1 - sponsored, 0 - not sponsored*


## How To
Download and extract the [training data](https://www.kaggle.com/c/dato-native/data) Part 0. Put it to the dir *./raw_data/0*

Run python script *truly_native.py*. Script will process raw html pages, do model training and perform prediction. in the end, classification report will be printed. 

**Attention!** Loading and processing raw html files can take around 30-40 minutes. Be patient;)

