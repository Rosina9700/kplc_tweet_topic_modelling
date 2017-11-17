# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
import pandas as pd
import got
import io


def get_tweets(query, filename):
	tweetCriteria = got.manager.TweetCriteria()
	tweetCriteria.since = '2017-10-26'
	tweetCriteria.until = '2017-11-16'
	tweetCriteria.maxTweets = 200000
	tweetCriteria.querySearch = query
	print 'Getting tweets for query: {}'.format(query)
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)
	print 'Savings tweets....'
	save_tweets(tweets, filename)
	return tweets

def save_tweets(tweets, filename):
	f = open('data_new/'+filename,'a')
	for t in tweets:
		string = '%s,%s,%d,%d,"%s",%s,%s,%s,%f,%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, float(t.id), t.permalink)
		f.write(string.encode('utf8')+'\n')
		f.flush()
	f.close()
	print '{} tweets added to file' .format(len(tweets))
	pass


if __name__ == '__main__':
	queries = ['@KenyaPowerAlert',
	      '@KenyaPowerAlerts', '@KenyaPower_Care',
	      '@KenyaPower_care', '@Kenya_Power',
	      '@Kenya_power', '@Kenyapower', '@KenyapowerAlert',
	      '@Kenyapower_care', '@kenyaPower', '@kenya_power', '@kenyapower', '@kenyapower0',
	      '@kenyapowerAlert', '@kenyapower_', '@kenyapower_Care',
	      '@kenyapower_care', '@kenyapoweralert', '@kenyapowercare', '@kplc', '@kplc_power', '#KPLC','#Kplc','#kplc', '#kenyapower',
	      '#KenyaPower','#kenya_power','#Kenya_Power', '#Kenya_power','#kenyapower_Care','#kenya_power_care','#Kenya_power_care','#Kenya_Power_care',
	      '#Kenya_Power_Care']
	# queries = ['@kplc', '@kplc_power', '#KPLC','#Kplc','#kplc', '#kenyapower',
 #  	      '#KenyaPower','#kenya_power','#Kenya_Power', '#Kenya_power','#kenyapower_Care','#kenya_power_care','#Kenya_power_care','#Kenya_Power_care',
 #  	      '#Kenya_Power_Care']
	for q in queries:
		if q[0] == '@':
			filename = 'tweets_at_'+q[1:]+'.csv'
		elif q[0] == '#':
			filename = 'tweets_hash_'+q[1:]+'.csv'
		f = codecs.open('data_new/'+filename, "w+", "utf-8")
		f.write('username,date,retweets,favorites,text,geo,mentions,hashtags')
		f.close()
		tweets = get_tweets(q, filename)
