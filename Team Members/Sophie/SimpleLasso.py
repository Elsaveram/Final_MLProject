# Testing engineer features Friday
%reload_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from LabelClass import LabelCountEncoder
from scipy.stats import skew
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from house import *
from config import *

def rmse_cv(model, x, y, k=5):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_log_error", cv = k))
    return(np.mean(rmse))

def plot_results(prediction):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, prediction, s=20)
    plt.title('Predicted vs. Actual')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
    plt.tight_layout()

def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))


del house
house = House('data/train.csv','data/test.csv')

# %% Clean data

house.cleanRP()
house.convert_types(HOUSE_CONFIG)

skewness = house.all.select_dtypes(exclude = ["object"]).apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
skewed_features

for i in range(0,len(skewed_features)):
    house.all[skewed_features[i]]=house.all[skewed_features[i]]+1
    house.all[skewed_features[i]]=house.all[skewed_features[i]].apply(np.log)

house.engineer_features(HOUSE_CONFIG)

# house.sg_ordinals()
# house.label_encode_engineer()

train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)

# house.all=pd.get_dummies(house.all,drop_first=True, dummy_na=False)

# house.train().sample(10)
# x=house.train().drop(['SalePrice','test'],axis=1)
# y=house.train().SalePrice
# x_train, x_test, y_train, y_test = train_test_split(x,y)

# x_train = x.as_matrix().astype(np.float)
# np.any(np.isnan(x_train))
# y_train = y.as_matrix().astype(np.float)

from sklearn.model_selection import GridSearchCV

alpha_100 =[{'alpha': np.logspace(-4, 2, 100)}]
lasso = linear_model.Lasso(alpha=1,normalize=True)
lasso.fit(X_train,Y_train) # fit data
lasso.predict(X_test)

para_search = GridSearchCV(estimator=lasso,param_grid=alpha_100,scoring='neg_mean_squared_log_error',cv=5)
para_search.fit(x,y)

rmsle(para_search.predict(x_test),y_test)

#R ^2 Coefficient
lasso.score(x_train,y_train)
rmse_cv(lasso,x_train,y_train)
rmsle(lasso.predict(x_test),y_test)

# %% Compare with Add features
# del house_sf
# house_sf=House('data/train.csv','data/test.csv')
#
# house_sf.cleanRP()
# house_sf.sg_ordinals()
#
# house_sf.sm_addFeatures_corr()
#
# use_columns=[x for x in house_sf.all.columns if house_sf.all[x].dtype == 'object' ]
# house_sf.all=pd.get_dummies(house_sf.all, drop_first=True, dummy_na=True)
# # house_sf.label_encode_engineer()
#
# x=house_sf.train().drop(['SalePrice','test'],axis=1)
# y=house_sf.train().SalePrice
# x_train, x_test, y_train, y_test = train_test_split(x,y)
#
# lasso = linear_model.Lasso(alpha=1,normalize=True) # create a lasso instance
# lasso.fit(x_train, y_train) # fit data
# lasso.coef_, lasso.intercept_ # print out the coefficients
# #R ^2 Coefficient
# lasso.score(x_test,y_test)
#
# # rmse_cv(lasso,x_train,y_train)
# rmsle(lasso.predict(x_test),y_test)
