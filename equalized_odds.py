'''HW4-Problem3'''

'''author@monica_yan'''


import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# read data
data=pd.read_table('adult-new.data',header=None,sep=',')
test=pd.read_table('adult-new.test',header=None,sep=',')
X=data.iloc[:,0:14]
X_continuous=X.iloc[:,[0,2,4,10,11,12]]
scaler = preprocessing.StandardScaler().fit(X_continuous)
X_cont=scaler.transform(X_continuous)
X_cont=pd.DataFrame(X_cont)
X_category=X.iloc[:,[1,3,5,6,7,8,9,13]]
x_cat=pd.get_dummies(X_category)
X_cont=pd.DataFrame(X_cont)
X_train = pd.concat([X_cont,x_cat], axis=1)
y=data.iloc[:,14]
y_tf=pd.get_dummies(y).iloc[:,1]
y_tf=y_tf.astype(int)
y_tf[y_tf==0]=-1
x_test=test.iloc[:,0:14]
x_test_con=x_test.iloc[:,[0,2,4,10,11,12]]
scaler = preprocessing.StandardScaler().fit(x_test_con)
X_test_con=scaler.transform(x_test_con)
X_test_con=pd.DataFrame(X_test_con)
x_test_category=x_test.iloc[:,[1,3,5,6,7,8,9,13]]
x_test_cat=pd.get_dummies(x_test_category)
x_test = pd.concat([X_test_con,x_test_cat], axis=1)
new_col=np.zeros(x_test.shape[0])
x_test.insert(loc=81, column=X_train.columns[81], value=new_col)
y_test=test.iloc[:,14]
y_test_tf=pd.get_dummies(y_test).iloc[:,1]
y_test_tf=y_test_tf.astype(int)
y_test_tf[y_test_tf==0]=-1

# model selection

# logostic regression
param_grid = {'C': [0.01, 0.1, 1, 10] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
clf.fit(X_train,y_tf)
print(clf.best_params_)
log=LogisticRegression(penalty='l2',C=0.1)
log.fit(X_train,y_tf)
y_predict=log.predict(x_test)
y_train_predict=log.predict(X_train)
print(sum(y_predict!=y_test_tf)/len(y_predict),'test error')
print(sum(y_train_predict!=y_tf)/len(y_train_predict),'train error')

# decision tree
param_grid = {'max_features': [0.3,0.4,0.5,0.6] }
dt = tree.DecisionTreeClassifier()
clf = GridSearchCV( BaggingClassifier(dt,max_samples=0.4),param_grid)
clf.fit(X_train,y_tf)
print(clf.best_params_)
dt = tree.DecisionTreeClassifier()

# bagging classifier
bagging = BaggingClassifier(dt,max_samples=0.3, max_features=0.6)
bagging.fit(X_train, y_tf)
y_predict=bagging.predict(x_test)
y_train_predict=bagging.predict(X_train)
print(sum(y_predict!=y_test_tf)/len(y_predict),'test error')
print(sum(y_train_predict!=y_tf)/len(y_train_predict),'train error')

# random forest
param_grid = {'n_estimators': [2,5,10,20,30] }
clf = GridSearchCV( RandomForestClassifier(),param_grid)
clf.fit(X_train,y_tf)
print(clf.best_params_)
rf = RandomForestClassifier(n_estimators=30)
rf.fit(X_train, y_tf)
y_predict=rf.predict(x_test)
y_train_predict=rf.predict(X_train)
print(sum(y_predict!=y_test_tf)/len(y_predict),'test error')
print(sum(y_train_predict!=y_tf)/len(y_train_predict),'train error')

# final predict
x_test_male=x_test.loc[x_test['9_ Female']==0,]
y_test_male=y_test_tf.loc[x_test['9_ Female']==0,]
x_test_female=x_test.loc[x_test['9_ Female']==1,]
y_test_female=y_test_tf.loc[x_test['9_ Female']==1,]

y_pred_male=log.predict(x_test_male)
y_pred_female=log.predict(x_test_female)
y_pred_male_bg=bagging.predict(x_test_male)
y_pred_female_bg=bagging.predict(x_test_female)
y_pred_male_rf=rf.predict(x_test_male)
y_pred_female_rf=rf.predict(x_test_female)

# report
def fp_fn(y_pred,y_true):
    TN,FP,FN,TP=confusion_matrix(y_true, y_pred).ravel()
    FN_rate=FN/(FN+TP)
    FP_rate=FP/(FP+TN)
    return(FN_rate,FP_rate)
 
print(fp_fn(y_pred_male,y_test_male),'log male')
print(fp_fn(y_pred_female,y_test_female),'log female')
print(fp_fn(y_pred_male_bg,y_test_male),'bg male')
print(fp_fn(y_pred_female_bg,y_test_female),'bg female')
print(fp_fn(y_pred_male_rf,y_test_male),'rf male')
print(fp_fn(y_pred_female_rf,y_test_female),'rf female')
