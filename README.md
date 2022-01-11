# Tweet-Sentiment-Extraction
Extract support phrases for sentiment labels
# Introduction
Sentiment Analysis can be defined as the process of analyzing text data and categorizing them into Positive, Negative, or Neutral sentiments. Sentiment Analysis is used in many cases like Social Media Monitoring, Customer service, Brand Monitoring, political campaigns, etc. Analyzing customer feedback such as social media conversations, product reviews, and survey responses allows companies to understand the customer’s emotions better which is becoming more essential to meet their needs.

Tweet => "My ridiculous dog is amazing." [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

# Business Problem
It is almost impossible to manually sort thousands of social media conversations, customer reviews, and surveys. The problem I am trying to solve here is part of this Kaggle competition. In this problem, we are given some text data along with their sentiment(positive/negative/neutral) and we need to find the phrases/words that best support the sentiment.

# Data Overview
The dataset used here is from the Kaggle competition Tweet Sentiment Extraction. The dataset used in this competition is from phrases from Figure Eight’s Data for Everyone platform.

It consists of two data files train.csv and test.csv, where there are 27481 rows in training data and 3534 rows in test data.

List of columns in the dataset

textID: unique id for each row of data

text: this column contains text data of the tweet.

sentiment: the sentiment of the text (positive/negative/neutral)

selected_text: phrases /words from the text that best supports the sentiment

# Performance Metric
The performance metric used in this problem is the word-level Jaccard score. The Jaccard Score or Jaccard Similarity is one of the statistics used in understanding the similarity between two sets.
