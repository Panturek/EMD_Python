import pandas as ps 
import numpy as np
import sys

filename = sys.argv[1]
res_file = 'predict_results.txt'
data = ps.read_csv(filename, usecols=['reviewText', 'score'])

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

data['reviewText'] = data['reviewText'].str.lower()
data['reviewText'] = data['reviewText'].str.replace(r'[^\w\s]', '')
for word in stopwords.words('english'):
    data['reviewText'] = data['reviewText'].str.replace(word, '')
data['score'] = data['score'].apply(lambda x : int(x) )
data = data.dropna()

from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), analyzer='char')

X = tfidf.fit_transform( data['reviewText'] )
y = data['score']

dummy = load('classifiers/dummy.joblib')
dummy_pred = dummy.predict(X)

with open(res_file, 'w') as f:
    sys.stdout = f
    print('Dummy')
    print(classification_report(y, dummy_pred))

svm = load('classifiers/svm.joblib')
svm_pred = svm.predict(X)

with open(res_file, 'a') as f:
    sys.stdout = f
    print('SVM')
    print(classification_report(y, svm_pred))

nbc = load('classifiers/tree.joblib')
nbc_pred = nbc.predict(X)

with open(res_file, 'a') as f:
    sys.stdout = f
    print('Tree')
    print(classification_report(csr_matrix.toarray(y), nbc_pred))