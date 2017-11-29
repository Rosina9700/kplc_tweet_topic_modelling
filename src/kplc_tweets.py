# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
import pandas as pd
import got
import io
import os

class Scrape_tweets(object):
	def __init__(self, start_date, end_date, folder_name):
		'''
		PARAMETERS:
		-------------
		start_date = 'yyyy-mm-dd'
		end_date = 'yyyy-mm-dd'
		folder_name = string
		'''
		self.start_date = start_date
		self.end_date = end_date
		self.folder_name = folder_name

	def save_tweets(self, tweets, filename):
		'''
		Save the scraped tweets in a .csv file of filename in self.folder_name
		PARAMETERS:
		------------
		tweets = output from function get_tweets, a list of dictionarys where each element is a tweet
		filename = string, giving the filename to save under
		'''
		f = open(self.folder_name+'/'+filename,'a')
		for t in tweets:
			string = '%s,%s,%d,%d,"%s",%s,%s,%s,%f,%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, float(t.id), t.permalink)
			f.write(string.encode('utf8')+'\n')
			f.flush()
		f.close()
		print '{} tweets added to file' .format(len(tweets))
		pass

	def get_tweets(self, query):
		'''
		Using the Get Old Tweets (got) libaries to scrape tweets and save them.
		PARAMETERS:
		------------
		query = string query to search for in the tweets. Most effective if this is a mention or hashtags
		RETURNS:
		------------
		list of dictionarys where each element is a tweet
		'''
		tweetCriteria = got.manager.TweetCriteria()
		tweetCriteria.since = self.start_date
		tweetCriteria.until = self.end_date
		tweetCriteria.maxTweets = 200000
		tweetCriteria.querySearch = query
		print 'Getting tweets for query: {}'.format(query)
		tweets = got.manager.TweetManager.getTweets(tweetCriteria)
		print 'Savings tweets....'
		self.save_tweets(tweets, filename)
		return tweets


if __name__ == '__main__':
	'''
	Run the file from the terminal with the name of the data folder to save the data to following the script name. This can take relative paths.
	A new .csv file will be made for each query, stating whether it is searching for a mention 'at' or a hashtag 'hash' followed by the query text.
	'''
	folder_name = sys.argv[1]
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

	queries = ['@KenyaPowerAlert',
	      '@KenyaPowerAlerts', '@KenyaPower_Care',
	      '@KenyaPower_care', '@Kenya_Power',
	      '@Kenya_power', '@Kenyapower', '@KenyapowerAlert',
	      '@Kenyapower_care', '@kenyaPower', '@kenya_power', '@kenyapower', '@kenyapower0',
	      '@kenyapowerAlert', '@kenyapower_', '@kenyapower_Care',
	      '@kenyapower_care', '@kenyapoweralert', '@kenyapowercare', '@kplc', '@kplc_power', '#KPLC','#Kplc','#kplc', '#kenyapower',
	      '#KenyaPower','#kenya_power','#Kenya_Power', '#Kenya_power','#kenyapower_Care','#kenya_power_care','#Kenya_power_care','#Kenya_Power_care',
	      '#Kenya_Power_Care']
	start_date = '2017-10-26'
	end_date = '2017-11-16'
	scraper = Scrape_tweets(start_date, end_date, folder_name)
	for q in queries:
		if q[0] == '@':
			filename = 'tweets_at_'+q[1:]+'.csv'
		elif q[0] == '#':
			filename = 'tweets_hash_'+q[1:]+'.csv'
		f = codecs.open(folder_name+'/'+filename, "w+", "utf-8")
		f.write('username,date,retweets,favorites,text,geo,mentions,hashtags')
		f.close()
		tweets = scraper.get_tweets(q)
		scraper.save_tweets(tweets, filename)
