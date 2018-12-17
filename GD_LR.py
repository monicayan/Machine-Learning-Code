
'''HW3-Problem3'''

'''authora@monicayan'''

from scipy.io import loadmat
import numpy as np
import pandas as pd
data=loadmat('hw3data.mat')

x_train=data['data']
y_train=data['labels']
x_train=np.insert(x_train,0,1,axis=1)

#global x_train
def log_odd(beta1):
    log_odd=np.exp(x_train.dot(beta1))/(1+np.exp(x_train.dot(beta1)))
    return log_odd
def cost(beta1):
    temp1=x_train * log_odd(beta1)[:, np.newaxis]
    result=np.cumsum(temp1-y_train*x_train,axis=0)[-1]
    result=result/x_train.shape[0]
    return result
def mle_vec(beta1):
    temp1=np.log(1+np.exp(x_train.dot(beta1)))-y_train.reshape(x_train.shape[0],)*x_train.dot(beta1)
    result=np.cumsum(temp1)[-1]
    result=result/x_train.shape[0]
    return result

beta=np.zeros(4)
mle=mle_vec(beta)
count=1
while mle>0.65064:
    step_size=1
    temp_mle1=mle_vec(beta-step_size*cost(beta))
    temp_mle2=mle_vec(beta)-(step_size/2)*np.dot(cost(beta).T,cost(beta))
    while(temp_mle1>temp_mle2):
        step_size=step_size/2
        temp_mle1=mle_vec(beta-step_size*cost(beta))
        temp_mle2=mle_vec(beta)-(step_size/2)*np.dot(cost(beta).T,cost(beta))
    beta=beta-step_size*cost(beta)
    mle=mle_vec(beta)
    count=count+1

print(count)
print(beta)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

x_train=data['data']
data1=pd.DataFrame(x_train)
print(data1.describe())

x_train1=(x_train)/np.std(x_train,axis=0)
x_train1=np.insert(x_train1,0,1,axis=1)

#global x_train
def log_odd1(beta1):
    log_odd=np.exp(x_train1.dot(beta1))/(1+np.exp(x_train1.dot(beta1)))
    return log_odd
def cost1(beta1):
    temp1=x_train1 * log_odd1(beta1)[:, np.newaxis]
    result=np.cumsum(temp1-y_train*x_train1,axis=0)[-1]
    result=result/x_train1.shape[0]
    return result
def mle_vec1(beta1):
    temp1=np.log(1+np.exp(x_train1.dot(beta1)))-y_train.reshape(x_train1.shape[0],)*x_train1.dot(beta1)
    result=np.cumsum(temp1)[-1]
    result=result/x_train1.shape[0]
    return result

beta=np.zeros(4)
mle=mle_vec1(beta)
count=1
while mle>0.65064:
    step_size=1
    temp_mle1=mle_vec1(beta-step_size*cost1(beta))
    temp_mle2=mle_vec1(beta)-(step_size/2)*np.dot(cost1(beta).T,cost1(beta))
    while(temp_mle1>temp_mle2):
        step_size=step_size/2
        temp_mle1=mle_vec1(beta-step_size*cost1(beta))
        temp_mle2=mle_vec1(beta)-(step_size/2)*np.dot(cost1(beta).T,cost1(beta))
    beta=beta-step_size*cost1(beta)
    mle=mle_vec1(beta)
    count=count+1

count

cutoff=round(x_train.shape[0]*0.8)

#part D

#preparing data for our algo
x_train2=x_train[0:cutoff]
x_train2=np.insert(x_train2,0,1,axis=1)
x_valid2=x_train[cutoff:]
x_valid2=np.insert(x_valid2,0,1,axis=1)

def valid_error(beta1,x_valid):
    y_valid=data['labels'][cutoff:].reshape(x_valid.shape[0],)
    y_predict=x_valid.dot(beta1)
    y_predict=y_predict>0
    y_predict=y_predict.astype(int)
    valid_result=(sum(y_predict!=y_valid)/len(y_valid))
    return valid_result

step_list=[2**x for x in range(15)]
print(step_list)

beta=np.zeros(4)
x_train=x_train2
y_train=data['labels'][0:cutoff]
mle=mle_vec(beta)
cv_error=[]
j=0
for i in range(10000):
    step_size=1
    temp_mle1=mle_vec(beta-step_size*cost(beta))
    temp_mle2=mle_vec(beta)-(step_size/2)*np.dot(cost(beta).T,cost(beta))
    while(temp_mle1>temp_mle2):
        step_size=step_size/2
        temp_mle1=mle_vec(beta-step_size*cost(beta))
        temp_mle2=mle_vec(beta)-(step_size/2)*np.dot(cost(beta).T,cost(beta))
    if i in step_list and i>16: 
        valid_e=valid_error(beta,x_valid2)
        cv_error.append(valid_e)
        if len(cv_error)>2:
            if cv_error[-1]>0.99*cv_error[-2]:
                print(cv_error[-1],'validation error')
                print(i,'step')
                print(mle,'objective value')
                break
    beta=beta-step_size*cost(beta)
    mle=mle_vec(beta)

#preparing data for transformation data
cutoff=round(x_train.shape[0]*0.8)
x_train=data['data']
x_valid_transformed=x_train[cutoff:]
x_train_transformed=x_train[0:cutoff]
x_train_transformed=(x_train_transformed)/np.std(x_train_transformed,axis=0)
x_train_transformed=np.insert(x_train_transformed,0,1,axis=1)
x_valid_transformed=(x_valid_transformed)/np.std(x_valid_transformed,axis=0)
x_valid_transformed=np.insert(x_valid_transformed,0,1,axis=1)

beta=np.zeros(4)
cv_error=[]
x_train1=x_train_transformed
y_train=data['labels'][0:cutoff]
mle=mle_vec1(beta)
for i in range(10000):
    step_size=1
    temp_mle1=mle_vec1(beta-step_size*cost1(beta))
    temp_mle2=mle_vec1(beta)-(step_size/2)*np.dot(cost1(beta).T,cost1(beta))
    while(temp_mle1>temp_mle2):
        step_size=step_size/2
        temp_mle1=mle_vec1(beta-step_size*cost1(beta))
        temp_mle2=mle_vec1(beta)-(step_size/2)*np.dot(cost1(beta).T,cost1(beta))
    if i in step_list: 
        valid_e=valid_error(beta,x_valid_transformed)
        cv_error.append(valid_e)
        if len(cv_error)>2:
            if cv_error[-1]>0.99*cv_error[-2] and i>16:
                print(cv_error[-1],'validation error')
                print(i,'step')
                print(mle,'objective value')
                break
    beta=beta-step_size*cost1(beta)
    mle=mle_vec1(beta)














