import math
import numpy as np
from pydoc import doc

 
from nltk.corpus import stopwords
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted


print("*" * 150)
print("*" * 150)
print("*" * 150)
print("*" * 150)
stop_words = stopwords.words('english')
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

files_name = natsorted(os.listdir('DocumentCollection'))

docs = []
for files in files_name:

    with open(f'DocumentCollection/{files}', 'r') as f:
        document = f.read()
    tokenized_documents = word_tokenize(document)
    terms = []
    for word in tokenized_documents:
        if word not in stop_words:
            terms.append(word)
    docs.append(terms)

document_number = 1
positional_index = {}

for document in docs:
    for positional, term in enumerate(document):
        if term in positional_index:
            positional_index[term][0] = positional_index[term][0] +1
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
            else:
                positional_index[term][1][document_number] = [positional]
        else:
            positional_index[term] = []
            positional_index[term].append(1)  
            positional_index[term].append({})    
            positional_index[term][1][document_number] = [positional]            
    document_number += 1        

print(positional_index)
print("*"*200)
 
def positional_index_query(qu):
    final_list = [[] for i in range(10)]
    for word in qu.split():
        for key in positional_index[word][1].keys():
             if final_list[key-1] != []:
                    if final_list[key-1][-1] == positional_index[word][1][key][0]-1:
                           final_list[key-1].append(positional_index[word][1][key][0])

             else:
                    final_list[key-1].append(positional_index[word][1][key][0])

    for position, list in enumerate(final_list, start=1):
       if len(list) == len(qu.split()):
           print(position)


if __name__ == '__main__':

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
    print('	Term Frequency(TF)	')
    print(termFreqs)
    print("*" * 150)
    ###############################3
    def weighted_termFreq(x):
        if x>0:
            return math.log(x)+1
        return 0


    for i in range(1, len(docs)+1):
        termFreqs['doc'+str(i)] = termFreqs['doc'+str(i)].apply(weighted_termFreq)
        print(' \n after apply	w tf(1+ log tf)	\n ')
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
   # print(DocFreq)
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
    print('\n	tf*idf \n')
    print(term_freq_invc_doc_fre)
    print("*" * 150)
############################################
    doc_length=pd.DataFrame()

    def get_docs_length(col):
        return np.sqrt(term_freq_invc_doc_fre[col].apply(lambda x:x**2).sum())

    for column in term_freq_invc_doc_fre:
        doc_length.loc[0,column+'_len']=get_docs_length(column)
    print("*"*150)
    print('\n doc_length \n ')
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
    print(' \n normalize_termFreq_idf \n')
    print(normalize_termFreq_idf)
    ###########################3
    def returned(q):
     query = pd.DataFrame(index =normalize_termFreq_idf.index)
     query['tf']=[ 1 if x in q.split() else 0 for x in list(normalize_termFreq_idf.index)] 
     def get_w_tf( x):
        try:
             return  math.log10(x)+1
        except:
            return 0
     query['w_tf']=query['tf'].apply(lambda x:get_w_tf( x))
     query['idf']=tfd['idf']*query['w_tf']
     query['tf_idf']=query['w_tf']*query['idf']
     query['norm']=0
     for i in range(len(query)):
            query['norm'].iloc[i]=float(query['idf'].iloc[i])/ math.sqrt(sum(query['idf'].values**2))
     print("\n\n\n query")
     print(query)
     product=normalize_termFreq_idf.multiply(query['w_tf'], axis=0)
     query['norm']=0
     for i in range(len(query)):
              query['norm'].iloc[i]=float(query['idf'].iloc[i])/ math.sqrt(sum(query['idf'].values**2))
     product2=product.multiply( query['norm'],axis=0)
     print("\n\nquery lenght")
     print(math.sqrt(sum(x**2 for x in query['idf'].loc[q.split()])) )
     print("\n\n\n products")
    
     math.sqrt(sum(x**2 for x in query['idf'].loc[q.split()]))
     scorse={}
     for col in product2.columns:
        if 0 in product2[col].loc[q.split()].values:
           pass
        else:
            scorse[col]=product2[col].sum()
     prod_res=product2[list(scorse.keys())].loc[q.split()]
     print(prod_res)
     print("\n\nsum of doc")
     print(prod_res.sum())

     final_score=sorted(scorse.items(),key=lambda x:x[1],reverse=True)
     print("\n\ncosine similary")
     for doc in final_score:
        print(doc,end=' ')
     print("\n\n returned docs\n\n\n")
     for doc in final_score:
        print(doc[0],end=' ')
    a=input("\n\nenter query ")
    returned(a)
   