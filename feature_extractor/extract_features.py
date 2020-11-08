import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re
from collections import Counter
import liwc
import math
import json


# Extract all

def extract_all_features(reviews,min_doc_freq=2):
    struct_extract(reviews)
    ugr_extract(reviews,min_doc_freq=min_doc_freq)
    galc_extract(reviews)
    liwc_extract(reviews)
    inq_extract(reviews)

# STRUCT
 
def struct_extract(reviews):

    """
    review list(str): list of sentences 
    """
    
    #initiate dataframe
    results = pd.DataFrame(reviews)
    results.columns = ['review']

    #define local funcs
    def avg_sent_length(string):
        sentences = string.split('.')
        sum_len = 0
        for s in sentences:
            sum_len += len(s)
        return sum_len/len(sentences)

    def per_of_q(string):
    
        num_q = string.count("?")
        new_string = string.replace("?",".")
        sentences = new_string.split(".")
        
        return num_q/len(sentences)
    
    #extract feats
    results['length'] = results.review.apply(lambda x: len(x))
    results['num_tokens'] = results.review.apply(lambda x: len(x.split(' ')))
    results['num_sentences'] = results.review.apply(lambda x: x.count('.'))
    results['avg_sent_len'] = results.review.apply(lambda x: avg_sent_length(x))
    results['num_exclm_mark'] = results.review.apply(lambda x: x.count('!'))
    results['ratio_q'] = results.review.apply(lambda x: per_of_q(x))
    
    #save file
    results.to_csv('results/struct_feats.csv')


# UGR 

def ugr_extract(reviews,min_doc_freq=0):

    #remove stop_words 
    reviews_clean = []
    stop_words = stopwords.words('english');
    stop_words += ["the","and","it"]

    for r in reviews:
        clean_r = r
        for w in r.split(' '):
            if w in stop_words: 
                clean_r = clean_r.replace(w,"");
        reviews_clean.append(clean_r)

    # calculate tf-idf
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(reviews_clean)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    tf_idf = pd.DataFrame(denselist, columns=feature_names)

    #remove infrequent words
    cols = []
    for c in tf_idf.columns:
        col = tf_idf[c]
        num_non_zer = 0
        for r in col:
            if r != 0:
                num_non_zer += 1
        if num_non_zer < min_doc_freq:
            cols.append(c)
    tf_idf_freq = tf_idf.drop(columns=cols)
    
    #save file
    tf_idf_freq.to_csv('results/tf_idf_freq.csv')


# GALC

def galc_extract(reviews):
    
    # read galc dictionary
    with open('helping/galc_dict.json') as json_file:
        galc_dict = json.load(json_file)
    
    #init dataframe
    galc_feature = pd.DataFrame(np.zeros((len(reviews),len(galc_dict))))
    galc_feature.columns = list(galc_dict.keys())

    def galc_vector_feature(review):
        ps = PorterStemmer()
        dic = dict.fromkeys(galc_dict.keys(),0)

        for w in review.split(' '):
            word = w.replace('.','')
            stemmed = ps.stem(word)

            for categ,words in galc_dict.items():
                if stemmed in words:
                    dic[categ] += 1

        return dic.values()

    for i,r in galc_feature.iterrows():
        galc_feature.iloc[i] = galc_vector_feature(reviews[i])

    #Save file
    galc_feature.to_csv('results/GALC_Features.csv')


# LIWC 

def liwc_extract(reviews):
    parse, category_names = liwc.load_token_parser('helping/LIWC2007_English100131.dic')

    # define helpers
    def tokenize(text):
        # you may want to use a smarter tokenizer
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    def liwc_features(text):

        dic = dict.fromkeys(category_names,0)

        gettysburg_tokens = tokenize(text)
        gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))

        for k,v in gettysburg_counts.items():
            dic[k] = v

        return dic.values()
    
    # init dataframe
    liwc_feature = pd.DataFrame(np.zeros((len(reviews),len(category_names))))
    liwc_feature.columns = category_names
    
    #extract feats
    for i,r in liwc_feature.iterrows():
        liwc_feature.iloc[i] = liwc_features(reviews[i])

    #save file
    liwc_feature.to_csv('results/LIWC_Features.csv')


# INQURIER 

def inq_extract(reviews):
    
    #read inq
    inq = pd.read_excel('helping/inquirerbasic.xls')
    inq_categs = list(inq.columns)
    
    #init dataframe
    inq_features = np.zeros((1,len(inq_categs)),dtype=int)

    #extract features 
    for review in reviews:
        inq_feat = dict.fromkeys(inq_categs,0)
        for w in review.split(' '):
            clean = w.strip().replace('.',"").replace("?",'').replace(",","").replace(";",'').upper()
            # if the word exists in the dictionary
            if len(inq[inq['Entry'] == clean]) > 0:
                row = inq[inq['Entry']==clean].to_dict()
                for k,v in row.items():
                    vv = list(v.values())[0]
                    if isinstance(vv,str):
                        inq_feat[k] += 1

        # convert the dict to one row features 
        inq_feat_row = np.array(list(inq_feat.values()),dtype=int).reshape((1,len(inq_categs)))

        #combine with big matrix
        inq_features = np.concatenate((inq_features,inq_feat_row),axis=0)
        
    
    # save file
    inq_features = pd.DataFrame(inq_features)
    inq_features.to_csv('results/inq_features.csv')
