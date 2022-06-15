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
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.stem.isri import ISRIStemmer #ar
from nltk.stem.porter import * #eng


def extract_classify_test(features,classes,kf=0):
    
    f_kbest_k = features
    
    if kf != 0:
        sel_k = SelectKBest(chi2, k=kf)
        f_kbest_k = sel_k.fit_transform(features, classes)
    
    ####### Build the model 
    
    #Create Labels and integer classes
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    le.fit(classes)
    print("Classes found : ", le.classes_)

    #Convert classes to integers for use with ML
    int_classes = le.transform(classes)
    print("\nClasses converted to integers :", int_classes)

    from sklearn.model_selection import train_test_split

    #Split as training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(f_kbest_k, int_classes,random_state=1,test_size=0.1)
    
    
    ####### Classify & Test
    
    cm_acc = {"NB":(),"SVM":(),"DT":(),"RF":(),"NN":()}
    
    from sklearn.naive_bayes import MultinomialNB
    classifier_1 = MultinomialNB().fit(xtrain, ytrain)
    print("NB")
    cm_acc["NB"] = test(classifier_1,xtest,ytest)

    from sklearn import svm
    classifier_2 = svm.SVC(kernel='linear').fit(xtrain, ytrain)
    print("SVM")
    cm_acc["SVM"] = test(classifier_2,xtest,ytest)

    from sklearn import tree
    clf_3 = tree.DecisionTreeClassifier().fit(xtrain, ytrain)
    print("Decision Tree")
    cm_acc["DT"] = test(clf_3,xtest,ytest)

    from sklearn.ensemble import RandomForestClassifier

    clf_4 = RandomForestClassifier(max_depth=2, random_state=0).fit(xtrain,ytrain)
    print("Random Forest")
    cm_acc["RF"] = test(clf_4,xtest,ytest)

    from sklearn.neural_network import MLPClassifier
    clf_5 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=0, max_iter=10000).fit(xtrain,ytrain)
    print("NN")
    cm_acc["NN"] = test(clf_5,xtest,ytest)
    
    return cm_acc
    
def replace(x):
    if x != "helpful":
        return "not_helpful"
    else:
        return "helpful"

def test(clf,xtest,ytest):
    from sklearn import metrics
    
    #Predict on test data
    predictions=clf.predict(xtest)
    
    print("Confusion Matrix : ")
    cm = metrics.confusion_matrix(ytest, predictions)
    print(cm)
    
    accuracy = metrics.accuracy_score(ytest, predictions)
    prc = metrics.precision_score(ytest , predictions)
    recall = metrics.recall_score(ytest , predictions)
    f1 = metrics.f1_score(ytest , predictions)
    
    dec = 3
    
    print("Acc: ",round(accuracy,dec)," Prec: ",round(prc,dec)," Recall: ",round(recall,dec)," F1:",round(f1,dec))
    
    #if predictions.sum() != 0:
    #    precision = metrics.precision_score(ytest , predictions)
    #else:dsd
    #    precision = 0.0
    #print("Precision:",precision)
    #
    #print("------------------------")
    
    
    return(cm,accuracy,prc,recall,f1)
    #return(cm,accuracy,precision)
    
####### Features methods 


def extract_tfidf(reviews):
        
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    #setup wordnet for lemmatization
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    from sklearn.feature_extraction.text import TfidfVectorizer

    #Custom tokenizer that will perform tokenization, stopword removal
    #and lemmatization
    def customtokenize(str):
        tokens=nltk.word_tokenize(str)

        #Replace special characters
        token_list2 = [word.replace("'", "") for word in tokens ]

        #Remove punctuations
        token_list3 = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, token_list2))

        #Convert to lower case
        token_list4=[word.lower() for word in token_list3 ]

        #remove stop words
        nostop = list(filter(lambda token: token not in stopwords.words('english'), token_list4))

        #lemmatized
        lemmatized=[lemmatizer.lemmatize(word) for word in nostop ]

        return lemmatized

    #Generate TFIDF matrix
    vectorizer = TfidfVectorizer(tokenizer=customtokenize,min_df=3)
    tfidf = vectorizer.fit_transform(reviews)

    print("\nSample feature names identified : ", vectorizer.get_feature_names()[:25])
    print("\nSize of TFIDF matrix : ",tfidf.shape)

    return (tfidf,vectorizer)
    
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
        for i,s in enumerate(sentences):
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
    
    #drop the review
    results = results.drop(columns=['review'])
    
    #scale
    results = results - results.min()
    results = results / results.max()
    results = results.fillna(0)
    
    #save file
    return results

# GALC

def galc_extract(reviews):
    
    # read galc dictionary
    with open('galc_dict.json') as json_file:
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
        
    
    #scale
    galc_feature = galc_feature - galc_feature.min()
    galc_feature = galc_feature / galc_feature.max()
    galc_feature = galc_feature.fillna(0)

    #Save file
    return galc_feature


# LIWC 

def liwc_extract(reviews):
    import liwc
    parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')

    # define helpers
    def tokenize(text):
        # you may want to use a smarter tokenizer
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    def liwc_features(text):
        
        dic = dict.fromkeys(category_names,0)

        gettysburg_tokens = tokenize(text)
        from collections import Counter
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
    
    #scale
    liwc_feature = liwc_feature - liwc_feature.min()
    liwc_feature = liwc_feature / liwc_feature.max()
    liwc_feature = liwc_feature.fillna(0)

    #save file
    return liwc_feature

# INQURIER 

def inq_extract(reviews):
    
    #read inq
    inq = pd.read_excel('inquirerbasic.xls')
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
    
    
    #scale
    inq_features = pd.DataFrame(inq_features)
    inq_features = inq_features - inq_features.min()
    inq_features = inq_features / inq_features.max()
    inq_features = inq_features.fillna(0)
    
    # save file
    return inq_features

# extract aspects

def extract_aspects(reviews,aspects):
    aspect_reviews = np.zeros((len(reviews),len(aspects)))
    for i,review in enumerate(reviews):
        for j,aspect in enumerate(aspects):
            #count the number of occurances         
            aspect_reviews[i][j] = review.count(aspect)
    return aspect_reviews

def extract_aspects_df(reviews,aspects):
    
    aspect_reviews = np.zeros((len(reviews),len(aspects)))
    
    for i,review in enumerate(reviews):
        for j,aspect in enumerate(aspects):
            #count the number of occurances    
            print(aspect)
            aspect_reviews[i][j] = review.count(aspect)
    
    #make df
    aspect_reviews = pd.DataFrame(aspect_reviews)
    aspect_reviews.columns = aspects
    
    return aspect_reviews

# find max accuracy  

def find_max_acc(cm_acc):
    accs = [a[1] for a in cm_acc.values()]
    return max(accs)

def find_max_prec(cm_acc_pr):
    prcs = [a[2] for a in cm_acc_pr.values()]
    return max(prcs)
    ml
# join features 

def join_features(features_list):
    features = pd.DataFrame(features_list[0])
    for i in range(1,len(features_list)):
        f_2 = pd.DataFrame(features_list[i])
        cols_to_use = f_2.columns.difference(features.columns)
        features = features.join(f_2[cols_to_use])
    return features

def join_features_df(dfs_features_list):
    features = dfs_features_list[0]
    for i in range(1,len(dfs_features_list)):
        f_2 = dfs_features_list[i]
        cols_to_use = f_2.columns.difference(features.columns)
        features = features.join(f_2[cols_to_use])
    return features
    
# combine not helpful 

def combine_not_helpful(classes):
    return classes.map(lambda x: "not_helpful" if x != "helpful" else "helpful")


# extract test with best k

def ex_with_best_k(f,c,ks):
    
    # find all ks
    
    cm_accs = []
    
    for i in range(1,ks):
        print("k = ---------- ",i)
        sel_k = SelectKBest(chi2, k=i)
        f_kbest_k = sel_k.fit_transform(f, c)
        cm_acc = extract_classify_test(f_kbest_k,c)
        cm_accs.append(cm_acc)
    
    # find max acc
    
    max_accs = []
    for k,cm_acc in enumerate(cm_accs):
        max_accs.append(find_max_acc(cm_acc))
    max_a = max(max_accs)
    print(max_a)
    
    #find max precision
    
    #max_prc = []
    #for k,cm_acc in enumerate(cm_accs):
    #    max_prc.append(find_max_prec(cm_acc))
    #max_p = max(max_prc)
    #print(max_p)

def extract_aspects_stem_df(reviews,aspects,lang="en"):
    
    aspect_reviews = np.zeros((len(reviews),len(aspects)))
    stmr = PorterStemmer()
    if lang == "ar":
        stmr = ISRIStemmer()
    
    for i,review in enumerate(reviews):
        for j,aspect in enumerate(aspects):
            
            # Stem aspect
            word = aspect
            word_s = stmr.stem(word)
            
            # Review stem words
            rev_list = pd.Series(review.split(" "))
            rev_list.map(lambda x: stmr.stem(x))
            rev_list = rev_list.to_list()
            
            # Count the number of occurances    
            aspect_reviews[i][j] = rev_list.count(word_s)
    
    #make df
    aspect_reviews = pd.DataFrame(aspect_reviews)
    aspect_reviews.columns = aspects
    
    return aspect_reviews


def top_words(vect,tf_idf,classes,k_f,filename):
    tf_idf_words = vect.get_feature_names()

    sel_k = SelectKBest(chi2, k=k_f)
    best_k = sel_k.fit_transform(tf_idf, classes)
    words_is = sel_k.get_support()
    all_scores = sel_k.scores_

    words_scores = []
    for i,t in enumerate(words_is):
        if t:
            words_scores.append((tf_idf_words[i],all_scores[i]))
            
    #sort
    words_scores.sort(key=lambda tup: -tup[1])
    
    #save
    t_2 = [(tp[0],round(tp[1],2)) for tp in words_scores]
    df = pd.DataFrame(t_2)
    df.to_csv(filename)
    
    return words_scores