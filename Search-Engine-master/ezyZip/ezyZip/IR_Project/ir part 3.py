import math
import numpy as np
from pydoc import doc

from nltk.corpus import webtext
from nltk.corpus import stopwords
import pandas as pd
if __name__ == '__main__':
    English_stops = set(stopwords.words('english'))
    text1 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/1.txt"
    text2 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/2.txt"
    text3 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/3.txt"
    text4 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/4.txt"
    text5 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/5.txt"
    text6 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/6.txt"
    text7 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/7.txt"
    text8 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/8.txt"
    text9 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/9.txt"
    text10 = "C://Users/DELL/Downloads/IR_Project/DocumentCollection/10.txt"


    text1 = webtext.raw(text1)
    text2 = webtext.raw(text2)
    text3 = webtext.raw(text3)
    text4 = webtext.raw(text4)
    text5 = webtext.raw(text5)
    text6 = webtext.raw(text6)
    text7 = webtext.raw(text7)
    text8 = webtext.raw(text8)
    text9 = webtext.raw(text9)
    text10 = webtext.raw(text10)

    t1 = [text1]
    t2 = [text2]
    t3 = [text3]
    t4 = [text4]
    t5 = [text5]
    t6 = [text6]
    t7 = [text7]
    t8 = [text8]
    t9 = [text9]
    t10=[text10]

    txt_1 = [word for word in t1 for word in word.split()]
    txt_2 = [word for word in t2 for word in word.split()]
    txt_3 = [word for word in t3 for word in word.split()]
    txt_4 = [word for word in t4 for word in word.split()]
    txt_5 = [word for word in t5 for word in word.split()]
    txt_6 = [word for word in t6 for word in word.split()]
    txt_7 = [word for word in t7 for word in word.split()]
    txt_8 = [word for word in t8 for word in word.split()]
    txt_9 = [word for word in t9 for word in word.split()]
    txt_10 = [word for word in t10 for word in word.split()]

    docs=[txt_1,txt_2,txt_3,txt_4,txt_5,txt_6,txt_7,txt_8,txt_9,txt_10]



    words=[]

    for doc in docs:
        for word in doc:
             words.append(word)
####################
    def term_freq(doc):
        foundWord=dict.fromkeys(words,0)
        for word in doc:
            foundWord[word]+=1
        return foundWord
##########################
    termFreqs=pd.DataFrame(term_freq(docs[0]).values(),index=term_freq(docs[0]).keys())

    for i in range(1,len(docs)):
        termFreqs[i]=term_freq(docs[i]).values()
    print("*"*150)
    termFreqs.columns=['doc'+str(i) for i in range(1,11)]
    print(termFreqs)
    print("*" * 150)
    ###############################3
    def weighted_termFreq(x):
        if x>0:
            return math.log(x)+1
        return 0


    for i in range(1, len(docs)+1):
        termFreqs['doc'+str(i)] = termFreqs['doc'+str(i)].apply(weighted_termFreq)
    print(termFreqs)
####################################
    DocFreq = []

    for t in range(len(termFreqs)):
        x = termFreqs.iloc[t]
        n = 0
        for i in range(10):
          if x[i] != 0:
             n += 1
        DocFreq.append(n)
    #print(DocFreq)
    print("*" * 150)
#######################################
    tfd=pd.DataFrame(columns=['df','idf'])
    for i in range(len(termFreqs)):
       freq=termFreqs.iloc[i].values.sum()
       termDc = DocFreq[i]
      # tfd.loc[i, 'freq'] = freq
       tfd.loc[i, 'df'] = termDc
      # tfd.loc[i, 'we_tf']=math.log10(1+freq)
       tfd.loc[i, 'idf'] = math.log10(float(10) / (float(termDc)))
      # tfd.loc[i, 'tf_idf']=math.log10(1+freq)* math.log10(float(10) / (float(termDc)))
    tfd.index=termFreqs.index
    print(tfd)
    print("*" * 150)
    """
    tfd = pd.DataFrame(columns=['freq', 'df', 'we_tf', 'idf'])
    for i in range(len(termFreqs)):
     freq = termFreqs.iloc[i].values.sum()
     termDc = DocFreq[i]
     tfd.loc[i, 'freq'] = freq
     tfd.loc[i, 'df'] = termDc
     tfd.loc[i, 'we_tf'] = math.log10(1 + freq)
     tfd.loc[i, 'idf'] = math.log10(float(10) / (float(termDc)))
# tfd.loc[i, 'tf_idf']=math.log10(1+freq)* math.log10(float(10) / (float(termDc)))
   tfd.index = termFreqs.index
   print(tfd)
   print("*" * 150)"""
################################################
    term_freq_invc_doc_fre=termFreqs.multiply(tfd['idf'],axis=0)
    print(term_freq_invc_doc_fre)
    print("*" * 150)
############################################
    doc_length=pd.DataFrame()

    def get_docs_length(col):
        return np.sqrt(term_freq_invc_doc_fre[col].apply(lambda x:x**2).sum())

    for column in term_freq_invc_doc_fre:
        doc_length.loc[0,column+'_len']=get_docs_length(column)
    print("*"*150)
    print(doc_length)
    ###################################
    normalize_termFreq_idf=pd.DataFrame()
    def get_normaliztion(col, x):
        try:
             return x / doc_length[col +'_len'].values[0]
        except:
            return 0
    for column in term_freq_invc_doc_fre.columns:
        normalize_termFreq_idf[column]=term_freq_invc_doc_fre[column].apply(lambda x:get_normaliztion(column,x))

    print(normalize_termFreq_idf)
    ###########################3



