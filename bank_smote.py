# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:35:55 2021

@author: sowmi
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


################Data Cleaning ###############
obj_df.drop(["LN_ID"],axis=1,inplace=True) 
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
#######################################################

#EXT_SCORE_1
#drop the column due to  more than 60% NaA values
obj_df.drop(["EXT_SCORE_1"],axis=1,inplace=True)


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


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
for feature in obj_df.columns[:]:
    print(feature,":",len(obj_df[feature].unique()),'labels')

obj_df['CONTRACT_TYPE']= le.fit_transform(obj_df['CONTRACT_TYPE']) 
obj_df['GENDER']= le.fit_transform(obj_df['GENDER']) 
obj_df['FAMILY_STATUS']= le.fit_transform(obj_df['FAMILY_STATUS']) 
obj_df['HOUSING_TYPE']= le.fit_transform(obj_df['HOUSING_TYPE']) 
obj_df['INCOME_TYPE']= le.fit_transform(obj_df['INCOME_TYPE']) 
obj_df['EDUCATION']= le.fit_transform(obj_df['EDUCATION']) 
obj_df['WEEKDAYS_APPLY']= le.fit_transform(obj_df['WEEKDAYS_APPLY']) 

len(obj_df["ORGANIZATION_TYPE"].unique())
obj_df["ORGANIZATION_TYPE"].isnull().sum()
obj_df.ORGANIZATION_TYPE
obj_df["ORGANIZATION_TYPE"] = obj_df["ORGANIZATION_TYPE"].replace("NA1","novalue")
obj_df["ORGANIZATION_TYPE"].value_counts()
obj_df['ORGANIZATION_TYPE']= le.fit_transform(obj_df['ORGANIZATION_TYPE']) 


obj_df.isnull().sum()
#################Feature Engineering######################
pip install sweetviz
import sweetviz as sv
report = sv.analyze(obj_df)

#display the report in html
report.show_html('report.html')

#############Train Test Split################
X=obj_df.drop(["TARGET"],axis=1)
X
y=obj_df['TARGET']
y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=10)



##############local outlier factor#####################################

# identify outliers in the training dataset
# evaluate model performance with outliers removed using local outlier factor
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
#Calculate valid and fraudulent cases
NotValid = obj_df[obj_df['TARGET'] == 1]
Valid = obj_df[obj_df['TARGET'] == 0]

outlier_fraction = len(NotValid) / float(len(Valid))
print('Outlier Fraction: {}'.format(outlier_fraction))

print('NotValid Cases: {}'.format(len(NotValid)))
print('Valid Cases: {}'.format(len(Valid)))

count_classes = pd.value_counts(obj_df['TARGET'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("NotValid class histogram")
plt.xlabel("TARGET")
plt.ylabel("Frequency")
plt.show()

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


state = 1

classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=300,max_samples=len(X),
                                       contamination = outlier_fraction,
                                       random_state = state,verbose =0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors =50,algorithm='auto',leaf_size=20, metric='minkowski',
                                              p=2, metric_params=None,contamination = outlier_fraction)
}


#Fitting the models

n_outliers = len(NotValid)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    if clf_name == "Local Outlier Factor":
        y_predict = clf.fit_predict(X)
        scores_predict = clf.negative_outlier_factor_
    
    else:
        clf.fit(X)
        scores_predict = clf.decision_function(X)
        y_predict = clf.predict(X)
    
    #Fit the results to the Class variable format
    y_predict[y_predict == 1] = 0
    y_predict[y_predict == -1] = 1
    
    n_errors = (y_predict != y).sum()
    
    #Run the classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_predict))
    print(classification_report(y, y_predict))
    
    
    
    
    
    
    

############################ Model Building  ####################################
    
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

from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(n_estimators=10).fit(X_train,y_train)
prediction=rf_classifier.predict(X_test)
y.value_counts()
rf= RandomForestClassifier(n_estimators=100,criterion='gini',max_features='sqrt',min_samples_leaf=10,verbose=2,random_state=100)
model=rf.fit(X_train, y_train)

y_pred = model.predict(X_test)

## Accuracy
accuracy_score(y_test, y_pred)
#83.6%
######confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test,y_pred))
#pre 0 - 93% 1 - 15%


##############################################
#####do not scale the data for logistice regression##############


##### Scaling the data #########
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)





######### LogisticRegressionClassifier######
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
grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=10, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Grid_accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#91%

# finding best estimator..#
grid_result.best_estimator_

lr= LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
                   
lr.fit(X,y)
lr_pred=lr.predict(X_test)

print(classification_report(y_test,lr_pred))
#Acc 89%   pre 0 - 96 ; 1 - 38%


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

###After Smote Logistic Regression####

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

## Logistic Regression With smote

%%time
lr = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'sag'] 
penalty = ['l2']  #he ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties
c_values = [100, 10, 1.0, 0.1, 0.01,.001] 
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train_res, y_train_res)

# summarize results
print("Grid_accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
##Gridacc= 68%

# finding best estimator..#
grid_result.best_estimator_

lr= LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
                   

lr.fit(X_train_res,y_train_res)

lr_pred = model.predict(X_test)

print(classification_report(y_test,lr_pred))
#acc = 89%
#precision = 0- 96%; 1- 38%
#recall = 0 - 93% ; 1 - 50%


lr_roc_auc = roc_auc_score(y_test, lr.predict(X_test)) 
print(lr_roc_auc )
#auc=0.676

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
  
print (lr_roc_auc) # .67








### XgBoost........###############################

import xgboost as xgb 

XGBC=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
              learning_rate=0.25, max_delta_step=0, max_depth=12,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1) 

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train) 
X_train_res.shape


model=XGBC.fit(X_train,y_train) ## smote

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors) #988
print("Accuracy:",accuracy_score(y_test, y_pred)) ## 91.76%..
print(classification_report(y_test,y_pred))
## pre 0--92%, 1--39% Recall 0- 100% ; 1 - 2%
#acc = 92%
## ROC Curve...

lr_roc_auc = roc_auc_score(y_test, XGBC.predict(X_test)) 
print(lr_roc_auc )
#auc=0.676

fpr, tpr, thresholds = roc_curve(y_test, XGBC.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='XGBoost Classifier(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
  
print (lr_roc_auc) # .67

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, lr_pred)
print(confusion_matrix)

### catboost..##
#pip install catboost
from catboost import CatBoostClassifier
cb=CatBoostClassifier(iterations=400,learning_rate=0.03,depth=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.4,random_state =50)

model=cb.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Accuracy
accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))
#Acc = 9290287789%
# pre 0 - 92% 1- 68%


lr_roc_auc = roc_auc_score(y_test, cb.predict(X_test)) 
print(lr_roc_auc )
#auc=0.676

fpr, tpr, thresholds = roc_curve(y_test, cb.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='CB Classifier(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 
  
print (lr_roc_auc) # .67





############################         Test Data        #############################################################

df=pd.read_csv("app_test.csv")


df.size
df.shape
df.columns
df.describe()
df.isnull().sum()
df.head()
df.dtypes

################Data Cleaning Test Data ###############

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


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
for feature in df.columns[:]:
    print(feature,":",len(df[feature].unique()),'labels')

df['CONTRACT_TYPE']= le.fit_transform(df['CONTRACT_TYPE']) 
df['GENDER']= le.fit_transform(df['GENDER']) 
df['FAMILY_STATUS']= le.fit_transform(df['FAMILY_STATUS']) 
df['HOUSING_TYPE']= le.fit_transform(df['HOUSING_TYPE']) 
df['INCOME_TYPE']= le.fit_transform(df['INCOME_TYPE']) 
df['EDUCATION']= le.fit_transform(df['EDUCATION']) 
df['WEEKDAYS_APPLY']= le.fit_transform(df['WEEKDAYS_APPLY']) 

len(df["ORGANIZATION_TYPE"].unique())
df["ORGANIZATION_TYPE"].isnull().sum()
df.ORGANIZATION_TYPE
df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("NA1","novalue")
df["ORGANIZATION_TYPE"].value_counts()
df['ORGANIZATION_TYPE']= le.fit_transform(df['ORGANIZATION_TYPE']) 



#################Feature Engineering######################
pip install sweetviz
import sweetviz as sv
report1 = sv.analyze(df)

#display the report in html
report1.show_html('report.html')

########### COmpareing train and test data

my_report = sv.compare([obj_df, "Training Data"], [df, "Test Data"])

my_report.show_html('myreport.html')


###########################Test data##############################
df.drop(["LN_ID"],axis=1,inplace=True) 
Z=df.drop(["TARGET"],axis=1)
test_target=df['TARGET']

###After Smote Logistic Regression GridCV####


lr_pred_test = model.predict(Z)

print(classification_report(test_target,lr_pred_test))



######## applying test data XGBoost  #############

y_pred_xg=model.predict(Z)


print(classification_report(test_target,y_pred_xg))

#############LogR#####

y_pred_lg = model.predict(Z)
print(classification_report(test_target,y_pred_lg))
#pre 0 - 94:  1 - 14
#Acc = 74%



######Catboost################
y_pred_cb=model.predict(Z)
print(classification_report(test_target,y_pred_cb))
#acc 92%
# pre 0 - 92% ; 1 - 31%





############################### Threashold Value to Maximize Accuracy################################


## Apply RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
ytrain_pred = rf_model.predict_proba(X_train)
print('RF train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = rf_model.predict_proba(X_test)
print('RF test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

ytrain_pred

##Apply Logistic Regression
from sklearn.linear_model import LogisticRegression
log_classifier=LogisticRegression()
log_classifier.fit(X_train, y_train)
ytrain_pred = log_classifier.predict_proba(X_train)
print('Logistic train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = log_classifier.predict_proba(X_test)
print('Logistic test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

#####Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada_classifier=AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
ytrain_pred = ada_classifier.predict_proba(X_train)
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = ada_classifier.predict_proba(X_test)
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))

##KNNClassifier 
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
ytrain_pred = knn_classifier.predict_proba(X_train)
print('KnnClassifier train roc-auc: {}'.format(roc_auc_score(y_train, ytrain_pred[:,1])))
ytest_pred = knn_classifier.predict_proba(X_test)
print('KnnClassifier test roc-auc: {}'.format(roc_auc_score(y_test, ytest_pred[:,1])))


#Now we will focus on selecting the best threshold for maximum accuracy
pred=[]
for model in [rf_model,log_classifier,ada_classifier,knn_classifier]:
    pred.append(pd.Series(model.predict_proba(X_test)[:,1]))
final_prediction=pd.concat(pred,axis=1).mean(axis=1)
print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_prediction)))

pd.concat(pred,axis=1)

final_prediction

fpr, tpr, thresholds = roc_curve(y_test, final_prediction)
thresholds

from sklearn.metrics import accuracy_score
accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(final_prediction>thres,1,0)
    accuracy_ls.append(accuracy_score(y_test, y_pred, normalize=True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                        axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
accuracy_ls.head()
accuracy_ls


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    

plot_roc_curve(fpr,tpr)    

###################################################################################


