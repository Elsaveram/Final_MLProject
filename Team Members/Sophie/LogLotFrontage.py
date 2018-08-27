# Log LotFrontage

# saving EDA figures

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

del house
house = House('data/train.csv','data/test.csv')
#%%
house.all.LotArea.min()
from sklearn import linear_model
ols = linear_model.LinearRegression()

my_ind=house.all[house.all['LotFrontage'].isnull()].index


y_train=house.all['LotFrontage'].loc[house.all['LotFrontage'].isnull()==False].values
x_train=house.all['LotArea'].loc[house.all['LotFrontage'].isnull()==False].values
x_train=np.log(x_train)
from math import exp
ols.fit(x_train.reshape(-1,1), y_train)   #### What happen if we remove the 'reshape' method?
print("beta_1: %.3f" %ols.coef_)
print("beta_0: %.3f" %ols.intercept_)

for i in my_ind:
    print(np.log(house.all['LotArea'].loc[i])*ols.coef_)
    house.all.loc[i,'LotFrontage']=((np.log(house.all.loc[i,'LotArea'])*ols.coef_)+ols.intercept_)[0]

# house.all['LotFrontage'].fillna(exp(),inplace=True)
plt.scatter(house.all['LotArea'],house.all['LotFrontage'])

# plt.scatter(house.all['LotArea'],house.all['LotFrontage'])
