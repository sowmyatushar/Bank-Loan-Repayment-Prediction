# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:35:55 2021

@author: sowmya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE  
import warnings
warnings.filterwarnings('ignore')


obj_df=pd.read_csv("app_train.csv")

obj_df.size
obj_df.shape
obj_df.columns
obj_df.describe()
obj_df.isnull().sum()
obj_df.head()
obj_df.dtypes


################EDA ###############

#treating nan value in ANNUITY

obj_df["ANNUITY"].isnull().sum()
def impute_nan(obj_df,variable,median):
    obj_df[variable+"_median"]=obj_df[variable].fillna(median)
    
median=obj_df.ANNUITY.median()
impute_nan(obj_df,'ANNUITY',median)

obj_df["ANNUITY_median"].isnull().sum()

import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(111)
obj_df['ANNUITY'].plot(kind='kde', ax=ax)
obj_df.ANNUITY_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

obj_df["ANNUITY"]=obj_df["ANNUITY_median"]
obj_df.drop(["ANNUITY_median"],axis=1,inplace=True)  ## drop the original

#treating nan values in PRICE

obj_df["PRICE"].isnull().sum()
obj_df['PRICE'].dropna().sample(obj_df['PRICE'].isnull().sum(),random_state=0)
obj_df[obj_df['PRICE'].isnull()].index
def impute_nan(obj_df,variable):
    obj_df[variable+"_random"]=obj_df[variable]
    ##It will have the random sample to fill the na
    random_sample=obj_df[variable].dropna().sample(obj_df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=obj_df[obj_df[variable].isnull()].index
    obj_df.loc[obj_df[variable].isnull(),variable+'_random']=random_sample



impute_nan(obj_df,"PRICE")
obj_df.head()
obj_df["PRICE_random"].isnull().sum()
#Visualisation
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(111)
obj_df['PRICE'].plot(kind='kde', ax=ax)
obj_df.PRICE_random.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

obj_df["PRICE"]=obj_df["PRICE_random"]
obj_df.drop(["PRICE_random"],axis=1,inplace=True)


###########################################################
#EXT_SCORE2
obj_df["EXT_SCORE_2"].isnull().sum()
obj_df['EXT_SCORE_2'].dropna().sample(obj_df['EXT_SCORE_2'].isnull().sum(),random_state=0)
obj_df[obj_df['EXT_SCORE_2'].isnull()].index
def impute_nan(obj_df,variable,median):
    obj_df[variable+"_median"]=obj_df[variable].fillna(median)
    obj_df[variable+"_random"]=obj_df[variable]
    ##It will have the random sample to fill the na
    random_sample=obj_df[variable].dropna().sample(obj_df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=obj_df[obj_df[variable].isnull()].index
    obj_df.loc[obj_df[variable].isnull(),variable+'_random']=random_sample

median=obj_df.EXT_SCORE_2.median()


impute_nan(obj_df,"EXT_SCORE_2",median)

obj_df.head()
obj_df["EXT_SCORE_2_random"].isnull().sum()
#Visualisation

import matplotlib.pyplot as plt
%matplotlib inline


fig = plt.figure()
ax = fig.add_subplot(111)
obj_df['EXT_SCORE_2'].plot(kind='kde', ax=ax)
#obj_df.EXT_SCORE_2_median.plot(kind='kde', ax=ax, color='red')
obj_df.EXT_SCORE_2_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

# From The Viz we the median & randome state both are showing same variance with original
# we can use either imputation so i used random state. 

obj_df["EXT_SCORE_2"]=obj_df["EXT_SCORE_2_random"]
obj_df.drop(["EXT_SCORE_2_random"],axis=1,inplace=True)
obj_df.drop(["EXT_SCORE_2_median"],axis=1,inplace=True)

#############
#EXT_SCORE_3
obj_df["EXT_SCORE_3"].isnull().sum()
obj_df['EXT_SCORE_3'].dropna().sample(obj_df['EXT_SCORE_3'].isnull().sum(),random_state=0)
def impute_nan(obj_df,variable,median):
    obj_df[variable+"_median"]=obj_df[variable].fillna(median)
    obj_df[variable+"_random"]=obj_df[variable]
    ##It will have the random sample to fill the na
    random_sample=obj_df[variable].dropna().sample(obj_df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=obj_df[obj_df[variable].isnull()].index
    obj_df.loc[obj_df[variable].isnull(),variable+'_random']=random_sample


median=obj_df.EXT_SCORE_3.median()
impute_nan(obj_df,"EXT_SCORE_3",median)

#visualizatoin
fig = plt.figure()
ax = fig.add_subplot(111)
obj_df['EXT_SCORE_3'].plot(kind='kde', ax=ax)
obj_df.EXT_SCORE_3_median.plot(kind='kde', ax=ax, color='red')
obj_df.EXT_SCORE_3_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

#from the Viz we can conlude : Median is showing share variance from the original
#so we use random state as the imputation method


obj_df["EXT_SCORE_3"]=obj_df["EXT_SCORE_3_random"]
obj_df.drop(["EXT_SCORE_3_random"],axis=1,inplace=True)
obj_df.drop(["EXT_SCORE_3_median"],axis=1,inplace=True)
obj_df.drop(["EXT_SCORE_1"],axis=1,inplace=True)
#######################################
#Converting -ve values to abs values

#DAYS_AGE
obj_df['DAYS_AGE'] = obj_df['DAYS_AGE'].abs()
#DAYS_WORK
obj_df['DAYS_WORK'] = obj_df['DAYS_WORK'].abs()
#DAYS_REGISTRATION
obj_df['DAYS_REGISTRATION'] = obj_df['DAYS_REGISTRATION'].abs()
#DAYS_ID_CHANGE
obj_df['DAYS_ID_CHANGE'] = obj_df['DAYS_ID_CHANGE'].abs()


#################Categorical data #####################

obj_df.drop(["INCOME_TYPE"],axis=1,inplace=True)

# Getting Dummies from all other categorical vars\n",
for col in obj_df.dtypes[obj_df.dtypes == "object"].index:
    for_dummy = obj_df.pop(col)
    obj_df = pd.concat([obj_df, pd.get_dummies(for_dummy, prefix=col)], axis=1)
obj_df.head()


################ TEST DATA CLEANING ######################

df=pd.read_csv("app_test.csv")
df.drop(["INCOME_TYPE"],axis=1,inplace=True)
df["PRICE"].isnull().sum()
df['PRICE'].dropna().sample(df['PRICE'].isnull().sum(),random_state=0)
df[df['PRICE'].isnull()].index
def impute_nan(df,variable):
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample



impute_nan(df,"PRICE")
df.head()
df["PRICE_random"].isnull().sum()
#Visualisation
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(111)
df['PRICE'].plot(kind='kde', ax=ax)
df.PRICE_random.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

df["PRICE"]=df["PRICE_random"]
df.drop(["PRICE_random"],axis=1,inplace=True)
#######################################################

#EXT_SCORE_1
#drop the column due to  more than 60% NaA values
df.drop(["EXT_SCORE_1"],axis=1,inplace=True)


###########################################################
#EXT_SCORE2
df["EXT_SCORE_2"].isnull().sum()
df['EXT_SCORE_2'].dropna().sample(df['EXT_SCORE_2'].isnull().sum(),random_state=0)
df[df['EXT_SCORE_2'].isnull()].index
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample

median=df.EXT_SCORE_2.median()


impute_nan(df,"EXT_SCORE_2",median)

df["EXT_SCORE_2_random"].isnull().sum()
#Visualisation

import matplotlib.pyplot as plt
%matplotlib inline


fig = plt.figure()
ax = fig.add_subplot(111)
df['EXT_SCORE_2'].plot(kind='kde', ax=ax)
#obj_df.EXT_SCORE_2_median.plot(kind='kde', ax=ax, color='red')
df.EXT_SCORE_2_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

# From The Viz we the median & randome state both are showing same variance with original
# we can use either imputation so i used random state. 

df["EXT_SCORE_2"]=df["EXT_SCORE_2_random"]
df.drop(["EXT_SCORE_2_random"],axis=1,inplace=True)

df.drop(["EXT_SCORE_2_median"],axis=1,inplace=True)

#############
#EXT_SCORE_3
df["EXT_SCORE_3"].isnull().sum()
df['EXT_SCORE_3'].dropna().sample(df['EXT_SCORE_3'].isnull().sum(),random_state=0)
df[df['EXT_SCORE_3'].isnull()].index
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample

median=df.EXT_SCORE_3.median()


impute_nan(df,"EXT_SCORE_3",median)

df["EXT_SCORE_3_random"].isnull().sum()
#Visualisation

import matplotlib.pyplot as plt
%matplotlib inline


fig = plt.figure()
ax = fig.add_subplot(111)
df['EXT_SCORE_3'].plot(kind='kde', ax=ax)
df.EXT_SCORE_3_median.plot(kind='kde', ax=ax, color='red')
df.EXT_SCORE_3_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

# From The Viz we the median & randome state both are showing same variance with original
# we can use either imputation so i used random state. 

df["EXT_SCORE_3"]=df["EXT_SCORE_3_random"]
df.drop(["EXT_SCORE_3_random"],axis=1,inplace=True)
df.drop(["EXT_SCORE_3_median"],axis=1,inplace=True)

#Converting -ve values to abs values

#DAYS_AGE
df['DAYS_AGE'] = df['DAYS_AGE'].abs()
#DAYS_WORK
df['DAYS_WORK'] = df['DAYS_WORK'].abs()
#DAYS_REGISTRATION
df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'].abs()
#DAYS_ID_CHANGE
df['DAYS_ID_CHANGE'] = df['DAYS_ID_CHANGE'].abs()


#################Categorical data #####################

# Getting Dummies from all other categorical vars\n",
for col in df.dtypes[df.dtypes == "object"].index:
    for_dummy = df.pop(col)
    df = pd.concat([df, pd.get_dummies(for_dummy, prefix=col)], axis=1)
df.head()


########################### Feature Seletion ######################

#EXT_SCORE_1
#drop the column due to  more than 60% Na values
#INCOME_TYPE
obj_df['INCOME_TYPE'].corr(obj_df['TARGET'])
obj_df.corr()



#################Feature Engineering######################
pip install sweetviz
import sweetviz as sv
report = sv.analyze(obj_df)
#display the report in html
report.show_html('report.html')
### COmpareing train and test data
my_report = sv.compare([obj_df, "Training Data"], [df, "Test Data"])
my_report.show_html('myreport.html')

#############Train Test Split################

X=obj_df.drop(["TARGET"],axis=1)
X
y=obj_df['TARGET']
y

####  Model to Find most significant features####
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
%matplotlib qt
model=ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(22).plot(kind='barh')
plt.show()

# INCOME_TYPE has correlatoin with TARGET 0.05 and also has high collinarity DAYS_WORK
#Drop the INCOME_TYPE column
obj_df.drop(["INCOME_TYPE"],axis=1,inplace=True)
df.drop(["INCOME_TYPE"],axis=1,inplace=True)

#############Train Test Split################
X=obj_df.drop(["TARGET"],axis=1)
X
y=obj_df['TARGET']
y

############## Model Building ####################   

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from collections import Counter

#####RandomforestClassifier#####

## spliting the data into 80:20 ratio...###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 888) 
smote=SMOTE()

smt = SMOTE(random_state=777)
X_train, y_train= smt.fit_resample(X_train, y_train)

X_train.shape
y_train.shape

print('Resampled dataset shape %s' % Counter(y_train))

rf= RandomForestClassifier(n_estimators=100,criterion='gini',max_features='sqrt',min_samples_leaf=10,verbose=2,random_state=100)

model=rf.fit(X_train, y_train)

y_pred = model.predict(X_test)

## Accuracy
accuracy_score(y_test, y_pred)
#91%
######confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test,y_pred))
#prec 0 -93% 1 - 57%


######### LogisticRegressionClassifier###### Before smote ##########
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from searchgrid import set_grid, make_grid_search

%%time
lr = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'sag'] 
penalty = ['l2']  #he ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties
c_values = [100, 10, 1.0, 0.1, 0.01,.001] 
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=10, scoring='precision',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Grid_accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#91%

# finding best estimator..#
grid_result.best_estimator_

lr= LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
                   
lr.fit(X,y)
lr_pred=lr.predict(X_test)

print(classification_report(y_test,lr_pred))
#Acc 92%   pre 0 - 92% ; 1 - 58%


## ROC Curve...

lr_roc_auc = roc_auc_score(y_test, lr.predict(X_test)) 
print(lr_roc_auc )

fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='logistic regression(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
  
print (lr_roc_auc) # .5


################# Smote Logistic Regression##########################

sm = SMOTE(random_state=444)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =122) 
X_train.shape
y_train.shape
X_train_res, y_train_res = sm.fit_resample(X_train, y_train) ## data increased by 62,025..#
X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape
from collections import Counter
print('Resampled dataset shape %s' % Counter(y_train_res))

%%time
lr = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'sag'] 
penalty = ['l2']  #he ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties
c_values = [100, 10, 1.0, 0.1, 0.01,.001] 
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=5, scoring='f1',error_score=0)
grid_result = grid_search.fit(X_train_res, y_train_res)

# summarize results
print("Grid_presicion: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
##Gridacc= 92%

# finding best estimator..#
grid_result.best_estimator_

lr= LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
                   

lr.fit(X_train_res,y_train_res)

lr_pred = model.predict(X_test)

print(classification_report(y_test,lr_pred))
#acc = 92%
#precision = 0- 96%; 1-77%



lr_roc_auc = roc_auc_score(y_test, lr.predict(X_test)) 
print(lr_roc_auc )
#auc=61%

fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
  
print (lr_roc_auc) # .61

#######################       XgBoost       ###############################

### Xgboost implementation gridsearchCV##
import xgboost as xgb 


## Hyper Parameter Optimization
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }

XGBC=xgb.XGBClassifier() 

random_search=RandomizedSearchCV(XGBC,param_distributions=params,n_iter=10,scoring='precision',n_jobs=-1,cv=5,verbose=1)

%%time
random_search.fit(X,y)

# finding best estimator..#
random_search.best_estimator_


XGBC=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.1,
              learning_rate=0.25, max_delta_step=0, max_depth=12,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1) 


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

model=XGBC.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors) #994


print("Accuracy:",accuracy_score(y_test, y_pred)) ## 91.96%..

print(classification_report(y_test,y_pred)) # acc = 92 pre 0 - 92% ; 1 - 39%






######### Catboost ##################
from catboost import CatBoostClassifier
cb=CatBoostClassifier()

parameters =params = {'depth': [4,7,8,9],
          'learning_rate' : [0.03, 0.05, 0.10],
         'l2_leaf_reg': [4,7,8,9],
         'iterations': [500]}

randm = RandomizedSearchCV(estimator=cb,param_distributions=parameters,scoring='precision' ,
                               cv = 3, n_iter = 10, n_jobs=-1)

%%time
randm.fit(X_train, y_train)

# finding best estimator..#
print(randm.best_estimator_)

randm.best_params_ ## Run this part only if your system config is good..##

cb=CatBoostClassifier(eval_metric="AUC",
                         depth=4, iterations=500, l2_leaf_reg=9, learning_rate= 0.03)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

%%time                            
model=cb.fit(X_train, y_train)
y_pred_cb = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred_cb)) ## 92%..


print(classification_report(y_test,y_pred_cb)) ### precision 0 --92, 1-- 57; 
#acc = 92%

## ROC Curve...

cb_roc_auc = roc_auc_score(y_test, cb.predict(X_test)) #print(rf_roc_auc ) 
fpr, tpr, thresholds = roc_curve(y_test, cb.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='CatBoost(area = %0.2f)' % cb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NEWS Classification')
plt.legend(loc="lower right")
plt.savefig('cb_ROC')
plt.show() 
  
print (cb_roc_auc) # .5




############################         Test Data        #############################################################


Z=df.drop(["TARGET"],axis=1)

test_target=df['TARGET']

###After Smote Logistic Regression###############

cb_pred_test = model.predict(Z)

print(classification_report(test_target,cb_pred_test))
#precision 1 - 16% 0 - 93% Acc = 84%


df = pd.DataFrame({'LN_ID':df['LN_ID'] ,'output': cb_pred_test})

final_outcome=df.to_csv("C:/Users/tussh//submission.csv", index=False)



