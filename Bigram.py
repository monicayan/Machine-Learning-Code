import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import math
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer
import gc
gc.collect()
data=pd.read_csv('/home/ec2-user/reviews_tr.csv')
test=pd.read_csv('/home/ec2-user/reviews_te.csv')

print(data.shape)
#train_data[i,0] label y
#train_data[i,1] feature
from sklearn.feature_extraction.text import CountVectorizer

#data=data.iloc[0:200000,]
data.loc[data['label']==0,'label']=-1
print(data.shape)
s=list(range(1000000))
import random
random.shuffle(s)
data=data.iloc[s,:]
vectorizer = CountVectorizer(ngram_range=(2,2))
train_vectors = vectorizer.fit_transform(data.iloc[:,1])
x_train=csr_matrix(train_vectors)
test_vectors=vectorizer.transform(test.iloc[:,1])
x_test=csr_matrix(test_vectors)
print(x_train.shape)


def perceptron(train_data):
    weight=np.zeros(x_train.shape[1])
    row_train=train_data.shape[0]
    for i in range((row_train)):
        if i<=(row_train/2):
            temp_fea=x_train[i,:]
            temp_1=temp_fea.dot(weight)*train_data.iloc[i,0]
            temp_1=int(temp_1)        
            if temp_1<=0:
                weight=weight+temp_fea.dot(train_data.iloc[i,0])
                weight=np.array(weight)[0]
                weight_total=weight
        else:
            print(i)
            temp_fea=x_train[i,:]
            temp_1=temp_fea.dot(weight)*train_data.iloc[i,0]
            temp_1=int(temp_1)
            if temp_1<=0:
                weight=weight+temp_fea.dot(train_data.iloc[i,0])
                weight=np.array(weight)[0]
                weight_total=weight_total+weight

    return weight_total
weight1=perceptron(data)/500000
print('train done')
print(max(weight1))
print(min(weight1))
label_y=x_test.dot(weight1)
label_y[label_y>0]=1
label_y[label_y<=0]=0
error_rate=(label_y!=test.iloc[:,0]).sum()/len(label_y)
print(error_rate,'test error rate')
label_y_train=x_train.dot(weight1)
label_y_train[label_y_train>0]=1
label_y_train[label_y_train<=0]=-1
train_error_rate=(label_y_train!=data.iloc[:,0]).sum()/len(label_y_train)
print(train_error_rate,'train_error rate')
feature=vectorizer.get_feature_names()
minimal_weight = [(feature[i]) for i in (weight1.argsort()[:10]) ]
maximal_weight= [(feature[i]) for i in (weight1.argsort()[-10:]) ]
minimal_weight.sort()
maximal_weight.sort()
print(minimal_weight)
print(maximal_weight)
