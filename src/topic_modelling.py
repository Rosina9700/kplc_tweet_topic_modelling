import pandas as pd
import numpy as np
import operator

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import string

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
#     tokenizer = RegexpTokenizer(r"[\w']+")
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

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print documents[doc_index]

def hand_label_topics(H, vocabulary):
    '''
    Print the most influential words of each latent topic, and prompt the user
    to label each topic. The user should use their humanness to figure out what
    each latent topic is capturing.
    '''
    hand_labels = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:5]
        print 'topic', i
        print '-->', ' '.join(vocabulary[top_five])
        label = raw_input('please label this topic: ')
        hand_labels.append(label)
        print
    return hand_labels

if __name__=='__main__':
    print 'read data...'
    data = pd.read_csv('../data_output/kplc_customer_tweets.csv')
    print 'vectorize text...'
    text = data['text'].values
    vect, vocab = build_text_vectorizer(text, use_stemmer=True, max_features=150, use_tfidf=False)
    vectorized = vect(text)
    print 'NMF...'
    nmf = NMF(n_components=3, max_iter=200, alpha=0.00001)
    W = nmf.fit_transform(vectorized)
    H = nmf.components_
    print 'examine results...'
    display_topics(H,W,vocab,text,5,5)
    hand_labels = hand_label_topics(H, vocab)
    print 'save results...'
    data['main_topic'] = W.argmax(axis=1)
    data.to_csv('../data_output/kplc_customer_with_topics.csv')
