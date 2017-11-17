import pandas as pd
from geopy.geocoders import Nominatim

df = pd.read_csv('data/kplc_tweets_kplc.csv')
df_1 = df[df['text'].str.contains(' in ')]

# use geopy to get locations and check that they are in kenya/
