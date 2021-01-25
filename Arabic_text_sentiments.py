# This code is to get text sentiments

import os
import csv
import codecs
from textblob import TextBlob
import re, string, unicodedata
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import re
import emoji
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)



df_path = "data/english_books_reviews.csv"
df = pd.read_csv(df_path)
df = df.dropna()
#print (list(df))
#print (df.head(1))
#print (df["rating"].value_counts())

print (df.shape)

# sentment analysis 
analyzer = SentimentIntensityAnalyzer()
positive_comt = 0
negative_comt = 0
neutral_comt = 0
count = 0

sentiment_list = []

for text in df['review_text']:
        count = count + 1
        vs = analyzer.polarity_scores(text)
        print("{:-<65} {}".format(text, str(vs)))
        if vs["compound"] >= 0.05: # positive
            positive_comt = positive_comt + 1
            sentiment_list.append("pos_Sentiment")
        elif vs["compound"] <= -0.05: # nigative
            negative_comt = negative_comt +1
            sentiment_list.append("neg_Sentiment")
        elif (vs["compound"] > -0.05)&(vs["compound"]< 0.05): # nutral
            neutral_comt = neutral_comt +1
            sentiment_list.append("neutral_Sentiment")
            
#add the sentiment_list to the dataframe as a new column             
df["Sentiment_Score"] = sentiment_list
df.to_csv("english_book_reviews_sent.csv")

print ("Total number of comments is ", count)
print ("Number of positive sentiment comments is ",positive_comt,"= ",positive_comt)
print ("Number of negative sentiment comments is ",negative_comt,"= ",negative_comt)
print ("Number of neutral sentiment comments is ",neutral_comt,"= ",neutral_comt)

print ("Number of positive sentiment comments is ",positive_comt,"= ",100*positive_comt/count,"%")
print ("Number of negative sentiment comments is ",negative_comt,"= ",100*negative_comt/count,"%")
print ("Number of neutral sentiment comments is ",neutral_comt,"= ",100*neutral_comt/count,"%")
