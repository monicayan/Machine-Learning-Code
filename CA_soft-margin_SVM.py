
'''HW3 - Problem 4'''


'''author@monicayan'''

import numpy as np
from scipy.io   import loadmat, savemat
from sklearn.preprocessing import scale

# Load hw3data.mat
hw3 = loadmat('hw3data.mat')

# Define x and y
train_data = hw3['data']
train_labels = np.array(hw3['labels']).astype("int")

# Standardization
train_data_sd = scale(train_data, with_mean=True, with_std=True) 

# change 0 to -1
train_labels[train_labels == 0] = -1

print('preprocessing done')

# Parameter
N = train_labels.shape[0]
C = 10/N

# initialization
a = np.zeros((N,1))

for t in range(2):
    for i in range(N):
        each_sum = 0
        sum_total = 0

        for j in range(N):
            if j != i:
                each_sum = 2 * train_labels[i] * a[j] * train_labels[j] * train_data_sd[i].dot(train_data_sd[j].T)
                sum_total = each_sum + sum_total
        
        condition = (1 - sum_total)/ (2 * np.square(train_labels[i]) * train_data_sd[i].dot(train_data_sd[i].T))

        if condition > C:
            a[i] = C
        elif condition < 0:
            a[i] = 0
        else:
            a[i] = condition

# objective function 
a_sum = 0
second_sum = 0
total_second_sum = 0
for i in range(N):
    a_sum = a[i] + a_sum
    for j in range(N):
        second_sum = train_labels[i] * train_labels[j] * train_data_sd[i].dot(train_data_sd[j].T) * a[i] * a[j]
        total_second_sum = second_sum + total_second_sum
object_value = a_sum - total_second_sum

# weight vector
w = np.zeros((3,1))

for j in range(3):
    each_sum = 0
    total_sum = 0
    for i in range(N):
        each_sum = a[i] * train_labels[i] * train_data_sd[i,j]
        total_sum = each_sum + total_sum
    w[j] = total_sum

print('Object Value', object_value)
print('Weight Vector', w.T)
