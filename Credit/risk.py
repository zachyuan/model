#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv("./LoanStats_2016Q3.csv",skiprows=1,low_memory=False)
df.info()
df.head(3)

df.ix[:4,:7]
df.drop('id',1,inplace=True)
df.drop('member_id',1,inplace=True)
df.int_rate = pd.Series(df.int_rate).str.replace('%', '').astype(float)
df.ix[:4,:7]

print(df.loan_amnt != df.funded_amnt).value_counts()
df.query('loan_amnt != funded_amnt').head(5)
df.dropna(axis=0, how='all',inplace=True)
df.info()
df.dropna(axis=1, how='all',inplace=True)
df.info()
df.ix[:5,8:15]

print(df.emp_title.value_counts().head())
print(df.emp_title.value_counts().tail())
df.emp_title.unique().shape
df.drop(['emp_title'],1, inplace=True)
df.ix[:5,8:15]

df.emp_length.value_counts()
df.replace('n/a', np.nan,inplace=True)
df.emp_length.fillna(value=0,inplace=True)
df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['emp_length'] = df['emp_length'].astype(int)
df.emp_length.value_counts()

df.verification_status.value_counts()

df.info()
df.columns
pd.unique(df['loan_status'].values.ravel())
for col in df.select_dtypes(include=['object']).columns:
    print("Column {} has {} unique instances".format( col, len(df[col].unique())) )
# 处理对象类型的缺失，unique
df.select_dtypes(include=['O']).describe().T.\
assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x))))

df.revol_util = pd.Series(df.revol_util).str.replace('%', '').astype(float)
# missing_pct
df.drop('desc',1,inplace=True)
df.drop('verification_status_joint',1,inplace=True)
df.drop('zip_code',1,inplace=True)
df.drop('addr_state',1,inplace=True)
df.drop('earliest_cr_line',1,inplace=True)
df.drop('revol_util',1,inplace=True)
df.drop('purpose',1,inplace=True)
df.drop('title',1,inplace=True)
df.drop('term',1,inplace=True)
df.drop('issue_d',1,inplace=True)
# df.drop('',1,inplace=True)
# 贷后相关的字段
df.drop(['out_prncp','out_prncp_inv','total_pymnt',
         'total_pymnt_inv','total_rec_prncp', 'grade', 'sub_grade'] ,1, inplace=True)
df.drop(['total_rec_int','total_rec_late_fee',
         'recoveries','collection_recovery_fee',
         'collection_recovery_fee' ],1, inplace=True)
df.drop(['last_pymnt_d','last_pymnt_amnt',
         'next_pymnt_d','last_credit_pull_d'],1, inplace=True)
df.drop(['policy_code'],1, inplace=True)

df.info()

df.ix[:5,:10]
df.ix[:5,10:21]

print(df.columns)
print(df.head(1).values)
df.info()

df.select_dtypes(include=['float']).describe().T.\
assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x))))

df.drop('annual_inc_joint',1,inplace=True)
df.drop('dti_joint',1,inplace=True)
df.drop('annual_inc_joint',1,inplace=True)
df.drop('dti_joint',1,inplace=True)

df['loan_status'].value_counts()

df.loan_status.replace('Fully Paid', int(1),inplace=True)
df.loan_status.replace('Current', int(1),inplace=True)
df.loan_status.replace('Late (16-30 days)', int(0),inplace=True)
df.loan_status.replace('Late (31-120 days)', int(0),inplace=True)
df.loan_status.replace('Charged Off', np.nan,inplace=True)
df.loan_status.replace('In Grace Period', np.nan,inplace=True)
df.loan_status.replace('Default', np.nan,inplace=True)
# df.loan_status.astype('int')
df.loan_status.value_counts()

# df.loan_status
df.dropna(subset=['loan_status'],inplace=True)

cor = df.corr()
cor.loc[:,:] = np.tril(cor, k=-1) # below main lower triangle of an array
cor = cor.stack()
cor[(cor > 0.55) | (cor < -0.55)]

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder

Y = df.loan_status
X = df.drop('loan_status',1,inplace=False)
print(Y.shape)
print(sum(Y))
X = pd.get_dummies(X)
print(X.columns)
print(X.head(1).values)
X.info()
X.fillna(0.0,inplace=True)
X.fillna(0,inplace=True)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=123)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())

# param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
#               'max_depth': [1,2,3,4],
#               'min_samples_split': [50,100,200,400],
#               'n_estimators': [100,200,400,800]
#               }

param_grid = {'learning_rate': [0.1],
              'max_depth': [2],
              'min_samples_split': [50,100],
              'n_estimators': [100,200]
              }
# param_grid = {'learning_rate': [0.1],
#               'max_depth': [4],
#               'min_samples_leaf': [3],
#               'max_features': [1.0],
#               }

est = GridSearchCV(ensemble.GradientBoostingRegressor(),
                   param_grid, n_jobs=4, refit=True)

est.fit(x_train, y_train)

best_params = est.best_params_
print(best_params)

%%time
est = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=300,
                                         learning_rate=0.1,max_depth=1, random_state=0,loss='ls').\
fit(x_train, y_train)

est.score(x_test,y_test)

%%time
est = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=100,
                                         learning_rate=0.1,max_depth=2, random_state=0,loss='ls').\
fit(x_train, y_train)

est.score(x_test,y_test)

def compute_ks(data):

    sorted_list = data.sort_values(['predict'], ascending=[True])

    total_bad = sorted_list['label'].sum(axis=None, skipna=None, level=None, numeric_only=None) / 3
    total_good = sorted_list.shape[0] - total_bad

    # print "total_bad = ", total_bad
    # print "total_good = ", total_good

    max_ks = 0.0
    good_count = 0.0
    bad_count = 0.0
    for index, row in sorted_list.iterrows():
        if row['label'] == 3:
            bad_count += 1.0
        else:
            good_count += 1.0

        val = bad_count/total_bad - good_count/total_good
        max_ks = max(max_ks, val)

    return max_ks

test_pd = pd.DataFrame()
test_pd['predict'] = est.predict(x_test)
test_pd['label'] = y_test
# df['predict'] = est.predict(x_test)
print compute_ks(test_pd[['label','predict']])

# Top Ten
feature_importance = est.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# XGBoost
clf2 = xgb.XGBClassifier(n_estimators=50, max_depth=1, 
                            learning_rate=0.01, subsample=0.8, colsample_bytree=0.3,scale_pos_weight=3.0, 
                             silent=True, nthread=-1, seed=0, missing=None,objective='binary:logistic', 
                             reg_alpha=1, reg_lambda=1, 
                             gamma=0, min_child_weight=1, 
                             max_delta_step=0,base_score=0.5)

clf2.fit(x_train, y_train)
print(clf2.score(x_test, y_test))
test_pd2 = pd.DataFrame()
test_pd2['predict'] = clf2.predict(x_test)
test_pd2['label'] = y_test
print(compute_ks(test_pd[['label','predict']]))
print(clf2.feature_importances_)
# Top Ten
feature_importance = clf2.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

# RFR
clf3 = RandomForestRegressor(n_jobs=-1, max_depth=10,random_state=0)
clf3.fit(x_train, y_train)
print clf3.score(x_test, y_test)
test_pd3 = pd.DataFrame()
test_pd3['predict'] = clf3.predict(x_test)
test_pd3['label'] = y_test
print compute_ks(test_pd[['label','predict']])
print clf3.feature_importances_
# Top Ten
feature_importance = clf3.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

# XTR
clf4 = ExtraTreesRegressor(n_jobs=-1, max_depth=10,random_state=0)
clf4.fit(x_train, y_train)
print clf4.score(x_test, y_test)
test_pd4 = pd.DataFrame()
test_pd4['predict'] = clf4.predict(x_test)
test_pd4['label'] = y_test
print compute_ks(test_pd[['label','predict']])
print clf4.feature_importances_
# Top Ten
feature_importance = clf4.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')









