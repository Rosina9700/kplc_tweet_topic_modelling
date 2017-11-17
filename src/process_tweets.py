import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def get_filenames(path):
    filenames = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            filenames.append(os.path.join(path,file))
    return filenames

def read_tweets(path):
    filenames = get_filenames(path)
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        df = df.iloc[:,:8]
        df.columns = ['username','date','retweets','favorites','text','geo','mentions','hashtags']
        dfs.append(df)
    return dfs

def date_check(dfs, last_start_date):
    dfs_to_use = []
    for df in dfs:
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].min() < pd.to_datetime(last_start_date):
            df = df[df['date'] < pd.to_datetime('2017-10-20')]
            dfs_to_use.append(df)
    return dfs

def clean_data(dfs, date_check=False):
    if date_check:
        dfs = date_check(dfs,'2016/11/03')
    data = dfs[0].append(dfs[1:])
    data = data[(data['text'].str.len()<=140)&(data['text'].str.len()>0)]
    data.drop_duplicates(['date','text'],inplace=True)
    data = data[data['username']!='kpc']
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['hour'] = data['date'].dt.hour
    data['date2'] = data['date'].dt.date
    return data

def split_company_customer(data):
    data_company= data[data['username'].isin(['KenyaPower','KenGenKenya'])]
    data_customer = data[~data['username'].isin(['KenyaPower','KenGenKenya'])]
    return data_company, data_customer

if __name__=='__main__':
    print 'reading data...'
    tweets = read_tweets('../data/')
    print 'cleaning data...'
    tweets = clean_data(tweets, date_check=True)
    tweets.to_csv('../data_output/kplc_tweets.csv')
    print 'splitting data...'
    company, customer = split_company_customer(tweets)
    company.to_csv('../data_output/kplc_company_tweets.csv')
    customer.to_csv('../data_output/kplc_customer_tweets.csv')
