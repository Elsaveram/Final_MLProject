# House for Sunanda's Interaction features
# %%
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

def sctplot(x,y,i):

    plt.scatter(x,y)
    plt.title('SalePrice vs ' + str(i))
    return plt.show()

# %% Clean data

house.cleanRP()

# house.engineer_features(HOUSE_CONFIG)

house.sg_ordinals()
house.label_encode_engineer()

house.sm_addFeatures()

old_features=['OverallQual','OverallCond', 'GarageQual', 'GarageCond', 'ExterQual','ExterCond',
'KitchenAbvGr', 'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageArea', 'GarageQual',
'PoolArea', 'PoolQC', 'ExterQual', 'ExterCond', 'PoolArea', 'PoolQC',
 'BsmtFullBath', 'BsmtHalfBath','FullBath', 'HalfBath',
'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
'3SsnPorch', 'OpenPorchSF','EnclosedPorch', 'ScreenPorch']
new_features=['OverallGrade', 'GarageGrade', 'ExterGrade', 'KitchenScore', 'FireplaceScore','PoolScore', 'TotalBath', 'AllSF', 'AllPorchSF']

for feat in old_features:
    corr_=house.all[feat].corr(house.all['SalePrice'])
    print(str(feat)+str(corr_))
for new in new_features:
    corr_=house.all[new].corr(house.all['SalePrice'])
    print(str(new)+str(corr_))



new_features=['OverallGrade', 'GarageGrade', 'ExterGrade', 'KitchenScore', 'FireplaceScore','PoolScore', 'TotalBath', 'AllSF', 'AllPorchSF']
compare_features=list(zip(new_features,old_features))
compare_features[0][0]

i=compare_features[1][0]
x=house.all[i]
y=house.all['SalePrice']
x.corr(y)
sctplot(x,y,i)

compare_features[8][1]
compare_features

for j in np.arange(0,8):
    for p in np.arange(0,2):
        i=compare_features[j][p]
        x=house.all[i]
        y=house.all['SalePrice']
        print(str(i)+str(x.corr(y)))
        sctplot(x,y,i)





# after we plot
house.all.drop(['OverallQual','OverallCond', 'GarageQual', 'GarageCond', 'ExterQual',
'ExterCond','KitchenAbvGr', 'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageArea', 'GarageQual',
'PoolArea', 'PoolQC', 'ExterQual', 'ExterCond', 'PoolArea', 'PoolQC', 'BsmtFullBath', 'BsmtHalfBath',
'FullBath', 'HalfBath', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'OpenPorchSF',
'EnclosedPorch', 'ScreenPorch'],axis = 1,inplace=True)



# show originals
# house.corr_matrix(house.train(),'SalePrice')
# # show the ones we care about:
# house.corr_matrix(house.train(),'SalePrice',cols_pair=['SalePrice','OverallQual','OverallCond'])
#
# house.sm_addFeatures()
#
# house.corr_matrix(house.train(), 'SalePrice',cols_pair=['SalePrice', 'OverallGrade', 'GarageGrade', 'ExterGrade', 'KitchenScore',
#  'FireplaceScore', 'PoolScore', 'TotalBath', 'AllSF', 'AllPorchSF'])
