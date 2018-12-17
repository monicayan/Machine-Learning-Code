from scipy.io import loadmat
import numpy as np
import pandas as pd
ocr=loadmat('ocr.mat')
import matplotlib.pyplot as plt
from matplotlib import cm

def distance(test, train):
	dists = -2 * np.dot(test, np.transpose(train)) + np.sum(train**2,axis=1) + np.sum(test**2, axis=1)[:, np.newaxis]
	return dists

def predict_labels(train_data,train_labels,test_data):
	distance1=distance(test_data,train_data)
	pred_labels=train_labels[np.argmin(distance1,axis=1),]
	return pred_labels

import random
n_random=[1000, 2000, 4000, 8000]
total_test_error=[]
total_train_error=[]
sd_test=[]
sd_train=[]
train=ocr['data'].astype('float')
test=ocr['testdata'].astype('float')
for j in n_random:
	temp_test_error_list=[]
	temp_train_error_list=[]
	for i in range(10):#10 times
		sel = random.sample(range(60000),j)
		train1=ocr['data'][sel].astype('float')
		trainlabels=ocr['labels'][sel]
		pre_labels=predict_labels(train1,trainlabels,test)
		result = pre_labels==ocr['testlabels']
		temp_test_error= (result == False).sum()/float(len(result))
		temp_test_error_list.append(temp_test_error)
		pre_train_labs=predict_labels(train1,trainlabels,train)
		result_train=pre_train_labs==ocr['labels']
		temp_train_error=(result_train== False).sum()/float(len(result_train))
		temp_train_error_list.append(temp_train_error)    
	error=np.mean(temp_test_error_list)
	std_test=np.std(temp_test_error_list)
	total_test_error.append(error)
	sd_test.append(std_test)
	error_train=np.mean(temp_train_error_list)
	std_train=np.std(temp_train_error_list)
	total_train_error.append(error_train)
	sd_train.append(std_train)


print(total_test_error)
print(total_train_error)
print(sd_test)

import matplotlib.pyplot as plt
plt.figure()
#plt.plot(n_random, total_test_error, label='Test')
plt.errorbar(n_random, total_test_error,yerr=sd_test)
plt.legend(loc='upper right')
plt.title('test error rate')
plt.xlabel('n size traning data')
plt.ylabel('test error rate')
plt.show()

plt.figure()
#plt.plot(n_random, total_test_error, label='Test')
plt.errorbar(n_random, total_train_error,yerr=sd_train)
plt.legend(loc='upper right')
plt.title('train error rate')
plt.xlabel('n size traning data')
plt.ylabel('train error rate')
plt.show()






