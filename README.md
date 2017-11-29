# Topic modelling for KPLC tweets
An analysis written in Python to investigate the latent topics in KPLC tweets from November 2016 to November 2017. Once topics have been defined, a Random Forest Classifer is used to classify new tweets.

## Aim
The aim of this model is to gain insight into the sentiment of Kenya Power and Lighting Company customers via their tweets. Power outages are persist on the Kenyan grid, much to their customers annoyance. In recent years, tech savvy Kenyans have chosen twitter as their platform of choice to communicate with KPLC.

## Results
Tweets from November 2016 to November 2017 were scraped and analyzed using topic modeling.

There seems to be 3 main topics:
1) General Power Outage Complaint
eg: "@KenyaPower no power, no power at mageche....Kenya power do something to
restore power...too much darkness..."
2) Temporal Reference
eg: "@KenyaPower Good morning no lights since yesterday morning.. Mt no
14106799464 Ruai Kamullu.. Ruai office is not helping been calling since"
3) Account/Token Help Request
"@KenyaPower hi please kindly update my current bill,as i partly paid 2 days ago
but still shows the same balance a/c is 2335150-03,please"

A RandomForestClassifier was then used to classify which topic the tweet belonged to (note that no hyperparameter tuning has been conducted)

On the initial dataset used for topic modelling, cross validated score is 92%, with the best accuracy for topic 1.
New tweets were scraped for November 2017 and classified using the RandomForestClassifier. Although we cannot quantitatively evaluate the performance of the classifier on new data, we can qualitatively. The following are random examples of each topic from the classification on the new unseen dataset. Overall these are aligned with what is expected.
1) General Power Outage Complaint example:
'@KenyaPower_Care no power in Ongata Rongai what is the issue Kindly assist..
Regards.. @KenyaPower @KenyaPowerAlert'
2) Temporal Reference
'We have not had power since 8pm yesterday. I called was given this number
3590664 but nothing till today!'
3) Account/Token Help Request
'@KenyaPower_Care @KenyaPowerAlert How long will meter no. 3080956-01
remain in darkness. It is now 3days. @ConsumersKenya'


## Details
The web scraping portion of this analysis relies heavily of Jefferson-Henriques [GetOldTweets-Python](https://github.com/Jefferson-Henrique/GetOldTweets-python) repository. Adapting this code for my purposes, I was able to gather tweets referencing KPLC from November 2016 to November 2017.

Non-Negative Matrix Factorization is then used to find the underlying latent topics. This requires some qualitative interpretation to distinguish clear and useful topics.

A RandomForest Classifier is then used to classify new incoming tweet into the relevant category. This is tested on new, unseen data from November 2017 with success.

## Files
The main source code files are found in the src folder. Code in development are found in the development folder.
**src folder**
- kplc_tweets.py: scripts to scrape tweets. Call from the terminal with the folder location to store tweets as the first arguments eg: python kplc_tweets.py ../data1
- process_tweets.py: script to clean the tweets and separate out company and customer tweets. Cleaned data is saved into a data_output folder.
- topic_modelling.py; script to vectorize the tweets and conduct the topic modelling. This requires some human intervention to interpret the topics and decide on correct number of topics. The terminal will print out examples of each topic as well as the option to hand label the topics for your convenience. The tweets are saved again with a column corresponding to the topic.
- classify_new_tweets.py: script to train a classifying algorithm on the tweets used in topic modelling. terminal will print out cross validated accuracy and test f1 and accuracy for each category. This also tests the algorithm on new unseen data. 

## To Do list/ Improvements
- Restructure the text vectorizer so that it can be exported as pkl during topic modelling and reloaded in classification
- Split the data up into 6 month chunks and see how the topics change over time
- Set up an AWS instance to 1) run the webscraping script and scrape at the end of every week, 2) rerun the topic modelling for the last 6 months of data every month and 3) host a static website showing how the topics change over time.
- Research further into geopy to see if we can exact locational references from the tweets.

## Prerequisites
Written in Python 2.x

The GetOldTweets package assumes using Python 2.x. Expected package dependencies are listed in the "requirements.txt" file for PIP, you need to run the following command to get dependencies:
```
pip install -r requirements.txt
```
Natural Language Processing package NLTK used for topic modelling

Scikit-learn RandomForestClassifier used for classification
