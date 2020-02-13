# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
from ukrainian_stemmer import UkrainianStemmer # Stemmer for Ukrainian language
from sklearn.model_selection import train_test_split

def ua_tokenizer(text,ua_stemmer=True,stop_words=[]):
    """ Tokenizer for Ukrainian language, returns only alphabetic tokens. 
    
    Keyword arguments:
    text -- text for tokenize 
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
        
    """
    tokenized_list=[]
    text=re.sub(r"""['’"`�]""", '', text)
    text=re.sub(r"""([0-9])([\u0400-\u04FF]|[A-z])""", r"\1 \2", text)
    text=re.sub(r"""([\u0400-\u04FF]|[A-z])([0-9])""", r"\1 \2", text)
    text=re.sub(r"""[\-.,:+*/_]""", ' ', text)
    for word in nltk.word_tokenize(text): 
        if word.isalpha():
            word=word.lower() 
            if ua_stemmer is True:      
                word=UkrainianStemmer(word).stem_word()
            if word not in stop_words:
                tokenized_list.append(word) 
    return tokenized_list

def FreqDist_ngrams(text,n=1,ua_stemmer=True,stop_words=[]):
    """ Return the FreqDist from a text. 
    
    Keyword arguments:
    text -- the source text to be converted into ngrams
    n -- the degree of the ngrams
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
        
    """
    
    FreqDist_data=nltk.ngrams(ua_tokenizer(text,ua_stemmer=ua_stemmer,stop_words=stop_words),n)
    FreqDist_data=nltk.FreqDist(FreqDist_data)
    
    return FreqDist_data


def FreqDist_info(FreqDist_data,most_common=50):
    """ show informatione aboute FreqDist data. 
    
    Keyword arguments:
    FreqDist_data -- FreqDist data
    n -- the degree of the ngrams
    most_common -- number of most common tokes (default 50)
        
    """
    print ("Об'єкт: ", FreqDist_data)
    #print ('Гапакси: ',FreqDist_data.hapaxes)
    print ('Найбільш уживані токени: ',FreqDist_data.most_common(most_common))
    FreqDist_data.plot (most_common, cumulative = True)
    
    

def text_to_FreqDist_ngrams_for_range(text,n_start=1,n_end=1,most_common=50,ua_stemmer=True,stop_words=[]):
    """ Return the list of FreqDist from a text wich include from n_start to n_end - grams. 
    
    Keyword arguments:
    text -- the source text to be converted into ngrams
    n_start -- the degree of the ngrams for begin
    n_end -- the degree of the ngrams for end
    most_common -- number of most common tokes for each n(default 50)
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
        
    """
    list_of_FreqDist=[]
    for n in range(n_start,n_end+1):
        print ('Розряд n: ',n)
        FreqDist_data=text_to_FreqDist_ngrams(text,n=n,ua_stemmer=ua_stemmer,stop_words=stop_words)    
        FreqDist_data_info(FreqDist_data,most_common=most_common)
        list_of_FreqDist.append(FreqDist_data)
    
    return list_of_FreqDist
    
def dataframe_most_common(dataframe,columns_list=[],list_len=2000,n=1,ua_stemmer=True,stop_words=[]):
    """ Return the list of most common tokens in FreqDist. 
    
    Keyword arguments:
    FreqDist_data -- FreqDist data
    list_len -- number of most common tokes (default 50)
        
    """
    data=''
    for column in columns_list:
        data+=dataframe[column].str.cat(sep=' ')
        data+=' '
    data=FreqDist_ngrams(data,n=n,ua_stemmer=ua_stemmer,stop_words=stop_words)
    FreqDist_info(data)
    data=data.most_common(list_len)
    data=[token[0] for token in data]   # FreqDist_data=[]                    
    
    #for ngram in data:
                          
    #    FreqDist_data.append(ngram[0])
    return data     

def dataframe_class_most_common(dataframe,columns_list=[], y_column='',list_len=2000,n=1,ua_stemmer=True,stop_words=[]):
    """ Return the list of most common tokens in FreqDist. 
    
    Keyword arguments:
    FreqDist_data -- FreqDist data
    list_len -- number of most common tokes (default 50)
        
    """
    y_column_set=set(dataframe[y_column])
    
    dataset=[]
    for y in y_column_set:
        #print(y)
        data=''
        dataframe_y=dataframe[dataframe[y_column]==y]
        for column in columns_list:
            data+=dataframe_y[column].str.cat(sep=' ')
            data+=' '
        data=FreqDist_ngrams(data,n=n,ua_stemmer=ua_stemmer,stop_words=stop_words)
        #FreqDist_info(data)
        data=data.most_common(list_len)
        data=[token[0] for token in data]   # FreqDist_data=[]   
        #print (data)
        dataset.extend(data)
    #for ngram in data:
                          
    #    FreqDist_data.append(ngram[0])
    return set(dataset)    


def bag_of_words(text,n,word_features):
    """ Return the dict of bag_of_words. 
    
    Keyword arguments:
    text -- 
    word_features -- 
        
    """
    text=FreqDist_ngrams(text,n=n)

    features={}
    for word in word_features:
        #print (word)
        features['contains({})'.format(word)]=(word in text)
        #print (word)
    return features


def data_split(dataframe,word_features,X_column, y_column,n=1,dev_test_size=0.0,test_size=0.33):
    
    featuresets=[]
    featuresets_y=[]
    for _,rows in dataframe.iterrows():
        featuresets.append([bag_of_words(rows[X_column],n=n,word_features=word_features), rows[y_column]])
        featuresets_y.append(rows[y_column])

    train_set,test_set,_,__=train_test_split(featuresets,featuresets_y,stratify=featuresets_y,test_size=(dev_test_size+test_size))
    
    return test_set,train_set
# %% [code]
