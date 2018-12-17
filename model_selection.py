
'''HW4-Problem1'''

'''author@monica_yan'''




'''RandomForestRegressor'''
import numpy as np

from time import time
from scipy.stats import randint as sp_randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# get some data
import scipy.io 
mat = scipy.io.loadmat('hw4data.mat')
data = mat['data']
labels = mat['labels']
quiz = mat['quiz']
X = data[:5000]
y = labels[:5000]

# confine data range
n = len(data)
n = int((3*n/4))

X_validation_train = data[:int(3*n/4)]
y_validation_train = labels[:int(3*n/4)]
X_validation_test = data[int(3*n/4)+1:]
y_validation_test = labels[int(3*n/4)+1:]
X_train = data[:n]
y_train = labels[:n]
X_test = data[n+1:]
y_test = labels[n+1:]

# build a classifier
clf_rfr = RandomForestRegressor(n_jobs=-1)


# use a full grid over all parameters
param_grid = {"max_depth": [None],
              "max_features": [64],
              "min_samples_split": [2],
              "min_samples_leaf": [6,10,20],
              "bootstrap": [True],
              "n_estimators": [150,200,300]}

# run grid search
grid_search_rfr = GridSearchCV(clf_rfr, param_grid=param_grid, n_jobs=-1)
start = time()
grid_search_rfr.fit(X, y.ravel())

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_rfr.cv_results_['params'])))

print(grid_search_rfr.best_estimator_)

# validation using training data
clf_rfr = grid_search_rfr.best_estimator_
X_train = data[:int(3*n/4)]
y_train = labels[:int(3*n/4)]
X_validation = data[int(3*n/4)+1:]
y_validation = labels[int(3*n/4)+1:]
clf_rfr.fit(X_train, y_train.ravel())
y_pred = clf_rfr.predict(X_validation)
print(mean_squared_error(y_pred, y_validation))

# Train & Test

clf_rfr = grid_search_rfr.best_estimator_
X_test = data[n+1:]
y_test = labels[n+1:]
X_train = data[:n]
y_train = labels[:n]
clf_rfr.fit(X_train, y_train.ravel())
y_pred = clf_rfr.predict(X_test)
print(mean_squared_error(y_pred, y_test))


'''Multi-layer Perceptron Regressor (Neural Network)'''

from sklearn.neural_network import MLPRegressor

# build a classifier
clf_mlp = MLPRegressor()


# use a full grid over all parameters
param_grid = {"hidden_layer_sizes": [(80,), (100,), (120,)],
              "activation": ['identity', 'logistic', 'relu'],
              "solver": ['sgd', 'adam'],
              "validation_fraction": [0.1],
              "max_iter": [150,200,250]}

# run grid search
grid_search_mlp = GridSearchCV(clf_MLP, param_grid=param_grid, n_jobs=-1)
start = time()
grid_search_mlp.fit(X, y.ravel())

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_mlp.cv_results_['params'])))

print(grid_search_mlp.best_estimator_)

# Validation
clf_mlp = grid_search_mlp.best_estimator_
clf_mlp.fit(X_validation_train, y_validation_train.ravel())
y_validation_pred = clf_mlp.predict(X_validation_test)
print(mean_squared_error(y_validation_pred, y_validation_test))

# Train & Test
clf_mlp_best = grid_search_mlp.best_estimator_
clf_mlp_best.fit(X_train, y_train.ravel())
y_pred = clf_mlp_best.predict(X_test)
print(mean_squared_error(y_pred, y_test))


'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression

# build a classifier
clf_lr = LogisticRegression(n_jobs = -1)


# use a full grid over all parameters
param_grid = {"C": [1.1,1.2,1.3],
              "penalty": ['l2'],
              "solver": ['lbfgs', 'liblinear', 'sag']}

# run grid search
grid_search_lr = GridSearchCV(clf_lr, param_grid=param_grid, n_jobs=-1)
start = time()
grid_search_lr.fit(X, y.ravel())

print("GridSearchCV for LogisticRegression took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_lr.cv_results_['params'])))

print(grid_search_lr.best_estimator_)

# Validation
clf_lr = grid_search_lr.best_estimator_
clf_lr.fit(X_validation_train, y_validation_train.ravel())
y_validation_pred = clf_lr.predict(X_validation_test)
print(mean_squared_error(y_validation_pred, y_validation_test))

# Train & Test
clf_lr_best = grid_search_lr.best_estimator_
clf_lr_best.fit(X_train, y_train.ravel())
y_pred = clf_lr_best.predict(X_test)
print(mean_squared_error(y_pred, y_test))


'''Conditional Probability Test'''
quiz = mat['quiz']
y_quiz_cond_pred = (clf_rfr.predict(quiz)+1)/2
y_quiz_cond_pred_01 = np.clip(y_quiz_cond_pred,0,1)
p = np.mean(y_quiz_cond_pred_01)
print(p)

 