
## Data loading
# %%
%reload_ext autoreload
%autoreload 2

from house import *
from config import *

del house
house = House('data/train.csv','data/test.csv')
#house_rp=House('data/train.csv','data/test.csv')
# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.special import boxcox1p
from sklearn import linear_model

house.cleanRP()
#house.cleanRP()
house.all['AllSF'] = house.all['GrLivArea'] + house.all['TotalBsmtSF'] #+ house.all['1stFlrSF'] + house.all['2ndFlrSF']
#plt.scatter(house.all['1stFlrSF'],house.all['SalePrice'])
#plt.scatter(house.all['GrLivArea'],house.all['SalePrice'])
#plt.scatter(house.all['TotalBsmtSF'],house.all['SalePrice'])
# %%
plt.scatter(house.all['AllSF'],house.all['SalePrice'],alpha=0.5)
plt.figure(figsize=(10,6))
plt.xlabel('Total SF')
plt.ylabel('Sale Price')
plt.savefig('SPVSallsf.png', transparent=True ,bbox_inches='tight' ,dpi=400,format='png')

#ED=house.all[house.all['Neighborhood']=='Edwards']
#len(ED[ED['BsmtFinType2']=='Unf'])
#%%

########feature eng.

#outliers
house.all=house.all.drop([1298,523])
# adding features
house.all['AllSF'].corr(house.all['SalePrice'])
#house.all.drop(['TotRmsAbvGrd','GarageArea'],axis=1)

house.convert_types(HOUSE_CONFIG)


#### Taking care of skewed variables:
# first recognize the skewed featurs:
skewness = house.all.select_dtypes(exclude = ["object"]).apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
skewed_features
# log transforming the skewed features to get normal dist.
for i in range(0,len(skewed_features)):
    house.all[skewed_features[i]]=house.all[skewed_features[i]]+1
    house.all[skewed_features[i]]=house.all[skewed_features[i]].apply(np.log)
### ordinal features.

house.engineer_features(HOUSE_CONFIG)

house.all['garage_new']=
house.all['GarageCond']
house.all['FireplaceQu']
house.all['garage_new']
##################################################
# simple lasso model
train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)


############### Lasso model
lasso = linear_model.Lasso(normalize=True) # create a lasso instance
lasso.fit(X_train, Y_train)# fit data
print("The determination of Lasso regression is: %.4f" %lasso.score(X_train, Y_train))

y_pred=lasso.predict(X_test)
np.sqrt(np.mean((np.log1p(y_pred) - np.log1p( Y_test))**2))


====================================
# recording the scores.

######### feature Engeneneering

#0.9181

#with feature 0.9279
#removing outliers 0.9288
#0.1226
#0.1224
======================================
#Grid search with lasso model


train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)
#Log - scailing the sale price.
Y_train=Y_train.apply(np.log)
lasso = linear_model.Lasso(normalize=True)
grid_param = [{'alpha': np.logspace(-4, 2, 100)}]
para_search_lasso = GridSearchCV(estimator=lasso, param_grid=grid_param, scoring='neg_mean_squared_error', cv=5, return_train_score=True)
para_search_lasso.fit(X_train, Y_train)
y_pred=para_search_lasso.predict(X_test)
np.exp(y_pred)
print(para_search_lasso.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(para_search_lasso.best_score_)))
np.sqrt(np.mean((np.log1p(np.exp(y_pred)) - np.log1p( Y_test))**2))

###### Creating the submission file for Kaggel
ID=pd.DataFrame(house.Id)['Id']
predict=pd.DataFrame(para_search_lasso.predict(test),columns=['SalePrice'])
predict=np.exp(predict)

predict['ID']=list(range(1461,2920))
predict=predict[['ID','SalePrice']]
predict.to_csv('predict10',index=False)
