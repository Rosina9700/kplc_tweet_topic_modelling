# Topic modelling for KPLC tweets
An analysis written in Python to investigate the latent topics in KPLC tweets from November 2016 to November 2017. Once topics have been defined, a Random Forest Classifer is used to classify new tweets. 

## Details
The web scraping portion of this analysis relies heavily of Jefferson-Henriques [GetOldTweets-Python](https://github.com/Jefferson-Henrique/GetOldTweets-python) repository. Adapting this code for my purposes, I was able to gather tweets referencing KPLC from November 2016 to November 2017. 

Non-Negative Matrix Factorization is then used to find the underlying latent topics. This requires some qualitative interpretation to distinguish clear and useful topics.

A RandomForest Classifier is then used to classify new incoming tweet into the relevant category. This is tested on new, unseen data from November 2017 with success.

## Prerequisites for GetOldTweets
The GetOldTweets package assumes using Python 2.x. Expected package dependencies are listed in the "requirements.txt" file for PIP, you need to run the following command to get dependencies:
```
pip install -r requirements.txt
```

## 

