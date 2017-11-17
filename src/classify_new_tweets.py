import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
from collections import Counter
import os
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import random
import operator

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

def read_training_data():
    df = pd.read_csv('data_output/kplc_customer_with_topics.csv')
    return df

def clean_new_tweets(data):
    data = data[(data['text'].str.len()<=140)&(data['text'].str.len()>0)]
    data.drop_duplicates(['date','text'],inplace=True)
    data = data[data['username']!='kpc']
    data = data[~data['username'].isin(['KenyaPower','KenGenKenya'])]
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['hour'] = data['date'].dt.hour
    data['date2'] = data['date'].dt.date
    data['retweeted'] = np.where(data['retweets']>0,1,0)
    data['retweeted'].sum()/float(len(data['retweeted']))
    return data

def read_new_tweets():
    filenames = []
    for file in os.listdir("data_new/"):
        if file.endswith(".csv"):
            filenames.append(os.path.join("data_new/", file))
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        df = df.iloc[:,:8]
        df.columns = ['username','date','retweets','favorites','text','geo','mentions','hashtags']
        dfs.append(df)

    data = dfs[0].append(dfs[1:])
    data = clean_new_tweets(data)
    return data

def build_text_vectorizer(contents, use_stemmer=False, max_features=None, use_tfidf=True):
    '''
    Build and return a **callable** for transforming text documents to vectors,
    as well as a vocabulary to map document-vector indices to words from the
    corpus. The vectorizer will be trained from the text documents in the
    `contents` argument. If `use_tfidf` is True, then the vectorizer will use
    the Tf-Idf algorithm, otherwise a Bag-of-Words vectorizer will be used.
    The text will be tokenized by words, and each word will be stemmed iff
    `use_stemmer` is True. If `max_features` is not None, then the vocabulary
    will be limited to the `max_features` most common words in the corpus.
    '''
    if use_tfidf==True:
        Vectorizer = TfidfVectorizer
    else:
        Vectorizer = CountVectorizer
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    stop_set = set(stopwords.words('english'))

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        punctuations = list(string.punctuation)
        stems = [s for s in stems if s not in punctuations]
        stems = [s.lower() for s in stems if s.isalpha()]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary


if __name__ == '__main__':
    print 'reading in data'
    training_data = read_training_data()
    new_tweets = read_new_tweets()

    print 'creating vectorizer on training data'
    training_text = training_data['text'].values
    vect, vocab = build_text_vectorizer(training_text, use_stemmer=True, max_features=150,use_tfidf=False)
    train_vectorized = vect(training_text)

    print 'vectorizing new tweets'
    new_text = new_tweets['text'].values
    new_vectorized = vect(new_text)

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
