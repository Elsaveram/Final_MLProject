# Take 2
# %%
%reload_ext autoreload
%autoreload 2

import numpy as np

from house import *
from config import *

del house
house = House('data/train.csv','data/test.csv')

# %%
# feature importance
house.cleanRP()

house.all['TotalSF'] = house.all['TotalBsmtSF'] + house.all['1stFlrSF'] + house.all['2ndFlrSF']
house.all.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)
house.all['Kitchen*Quality']=house.all['KitchenAbvGr']*house.all['KitchenQual']
house.all.drop(['KitchenAbvGr','KitchenQual'],axis=1,inplace=True)
house.all['Fireplaces*Quality']=house.all['Fireplaces']*house.all['FireplaceQu']
house.all.drop(['Fireplaces','FireplaceQu'],axis=1,inplace=True)

maybe_drop=['3SsnPorch', 'BsmtFinSF2']
house.all.drop(maybe_drop,axis=1,inplace=True)

perform_log=['BsmtFinSF1',
       'BsmtUnfSF', 'EnclosedPorch', 'GarageYrBlt', 'GrLivArea', 'HalfBath',
       'LotArea', 'LotFrontage', 'LowQualFinSF', 'MSSubClass', 'MasVnrArea',
        'OpenPorchSF', 'OverallCond', 'PoolArea', 'ScreenPorch',
       'TotRmsAbvGrd', 'WoodDeckSF', 'TotalSF']
house.sg_skewness(mut=0)
for feat in house.skewed_features:
    house.log_transform(house.train()[feat])


from scipy.special import boxcox1p
lam = 0.15
house.all[perform_log] = boxcox1p(house.all[perform_log],lam)
house.train()['SalePrice'] = boxcox1p(house.train()['SalePrice'],lam)


house.sg_ordinals()
house.label_encode_engineer()

house.sg_statsmodels()

house.sk_random_forest(num_est=500)

# house.model_prediction
