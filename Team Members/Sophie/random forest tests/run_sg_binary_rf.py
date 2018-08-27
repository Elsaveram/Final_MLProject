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


house.cleanRP()
house.sg_ordinals()
house.label_encode_engineer()

# house.all.sample(10)

fix_log=['GrLivArea','1stFlrSF','LotFrontage'] #'SalePrice',
fix_log_maybe=['BsmtUnfSF','BsmtFinSF1']
house.all[fix_log] = np.log1p(house.all[fix_log])
house.all['SalePrice'] = np.log1p(house.all['SalePrice'])

house.sg_statsmodels()

house.sk_random_forest(num_est=500)
bin_house=house
change_binary=['TotalBsmtSF','GarageArea','2ndFlrSF','MasVnrArea','WoodDeckSF','OpenPorchSF','PoolArea']
for var in change_binary:
    bin_house.all[var+'_B']=bin_house.all[var].apply(lambda x: 1 if x > 0 else 0)
    bin_house.all.drop([var],axis=1)
bin_house.all['YearBuilt'] = bin_house.all['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

bin_house.sk_random_forest(num_est=500)

house.all.loc[house.all.SalePrice>50000]
