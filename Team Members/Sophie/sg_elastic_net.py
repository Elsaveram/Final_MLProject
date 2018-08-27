# Run an Elastic Net:
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
def plot_results(prediction):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, prediction, s=20)
    plt.title('Predicted vs. Actual')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
    plt.tight_layout()

def rmse_cv(model, x, y, k=5):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_log_error", cv = k))
    return(np.mean(rmse))

def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))

# %% Clean data

house.cleanRP()

# house.all['TotalSF'] = house.all['TotalBsmtSF'] + house.all['1stFlrSF'] + house.all['2ndFlrSF']
# house.all.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)

perform_log=['BsmtFinSF1','BsmtUnfSF', 'EnclosedPorch', 'GarageYrBlt', 'GrLivArea',
    'HalfBath', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MSSubClass', 'MasVnrArea',
        'OpenPorchSF', 'OverallCond', 'PoolArea', 'ScreenPorch',
       'TotRmsAbvGrd', 'WoodDeckSF', 'TotalSF']
# house.all[perform_log] = np.log1p(house.all[perform_log]+1)
house.train()['SalePrice'] = np.log1p(house.train()['SalePrice']+1)

# house.engineer_features(HOUSE_CONFIG)

house.sg_ordinals()
house.label_encode_engineer()
# house.all.to_csv('SophiePipeline1.csv')

house.sm_addFeatures()

# %% Elastic Search
x=house.train().drop(['SalePrice','test'],axis=1)
y=house.train().SalePrice
x_train, x_test, y_train, y_test = train_test_split(x,y)


# #
# alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]
# l1_ratio=[.01, .1, .5, .9, .99]
# max_iter=5000

x_train.sample(10)
ENSTest = linear_model.ElasticNet(alpha = 1, l1_ratio = 0.5)
ENSTest.fit(x_train, y_train)
elast_pred = ENSTest.predict(x_test)
rmsle(y_pred=ENSTest.predict(x_train),y_test=y_train)
ENSTest.get_params()
dir(ENSTest)
plot_results(elast_pred)
print('RMSLE from Kaggle: '+str(rmsle(y_pred=elast_pred,y_test=y_test)))
cross_val_score(ENSTest,x_train,y_train,scoring='neg_mean_squared_log_error',cv=5)
rmse_cv(ENSTest, x_train, y_train)

x_train

# %%
#creating file for Kaggel
id=np.arange(1461,2920)
predict=pd.DataFrame(ENSTest.predict(house.test().drop(['SalePrice','test'],axis=1)),columns=['SalePrice'])
predict['ID']=np.arange(1461,2920)#list(range(1461,2920))
predict=predict[['ID','SalePrice']]
predict.to_csv('predict.csv',index=False)
