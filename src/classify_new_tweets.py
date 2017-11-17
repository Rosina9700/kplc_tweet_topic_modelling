import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import string
plt.style.use('ggplot')
# plt.style.use('fivethirtyeight')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from process_tweets import read_tweets, clean_data, split_company_customer
from topic_modelling import build_text_vectorizer


if __name__ == '__main__':
    print 'reading in data'
    training_data = pd.read_csv('../data_output/kplc_customer_with_topics.csv')
    new_tweets = read_tweets('../data_new/')
    new_tweets = clean_data(new_tweets)

    print 'creating vectorizer on training data'
    training_text = training_data['text'].values
    vect, vocab = build_text_vectorizer(training_text, use_stemmer=True, max_features=150,use_tfidf=False)
    train_vectorized = vect.transform(training_text).toarray()

    print 'vectorizing new tweets'
    new_text = new_tweets['text'].values
    new_vectorized = vect.transform(new_text).toarray()

    print 'Training RFC'
    X = train_vectorized
    y = training_data['main_topic'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    print 'Performance on training data'
    # performance on training data
    y_pred = rfc.predict(X_test)
    cv_accuracy = cross_val_score(rfc, X_train, y_train, cv=5)
    print 'cross validated accuracy: {}'.format(cv_accuracy.mean())
    for i in xrange(3):
        true = y_test == i
        pred = y_pred ==i
        print 'topic {} accuracy: {}'.format(i, accuracy_score(true, pred))
        print 'topic {} f1: {}'.format(i, f1_score(true, pred))

    print 'Performance on new tweets'
    # performance on new tweets
    y_pred_new = rfc.predict(new_vectorized)
    for i in xrange(3):
        true = y_pred_new == i
        topic_tweets = new_tweets.iloc[true,:]
        print 'topic {}'.format(i+1)
        print topic_tweets['text'].values[:5]
