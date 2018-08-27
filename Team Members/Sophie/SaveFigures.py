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
# %%
#save the missingness chart
house.missing_stats()

house.cleanRP()

house.log_transform(house.train()['SalePrice'],'SalePrice')

house.distribution_charts()

for column in house.all.columns: #['Electrical']:#
    if house.all[column].dtype in ['object', 'int64']:
        plt.figure()
        house.all.groupby([column,'test']).size().unstack().plot.bar(color=[(0.3,0.5,0.4,0.6),(0.3,0.9,0.4,0.6)])
        plt.savefig('Figures/DistributionCharts/'+str(column)+'.png',bbox_inches='tight' ,dpi=400,transparent=True,  format='png')

house.sale_price_charts()
house.all['GarageYrBlt'].min()
house.sg_ordinals()
house.label_encode_engineer()

house.sg_skewness(mut=0)
for feat in house.skewed_features:
    house.log_transform(house.train()[feat],feat)

plt.scatter(house.all['LotFrontage'],house.all['LotArea'])

plt.scatter(np.log1p(house.all['LotArea']),np.log1p(house.all['LotFrontage']))

# %%

col_missing=[name for name in house.all.columns if np.sum(house.all[name].isnull()) !=0]
col_missing.remove('SalePrice')
print(col_missing)
my_missing=['Alley', 'BsmtCond','BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath',
 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'Electrical', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu',
 'Functional', 'GarageArea', 'GarageCars', 'GarageCond','GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',
  'KitchenQual', 'LotFrontage', 'MSZoning', 'MasVnrArea', 'MasVnrType', 'MiscFeature', 'PoolQC', 'SaleType', 'TotalBsmtSF', 'Utilities']
heat=sns.heatmap(house.all[my_missing].isnull(),cbar=False)
fig=heat.get_figure()
plt.savefig('Figures/Missingness/HeatMapAll.png', transparent=True,bbox_inches='tight' ,dpi=400,format='png')

#%%
import missingno as msno
heat=msno.heatmap(house.all)
# sns_plot = sns.pairplot(df, hue='species', size=2.5)
fig=heat.get_figure()
# fig = sns_plot.get_figure()
fig.savefig("output.png")
fig.savefig('Figures/Missingness/HeatMapAll.png', bbox_inches='tight' ,dpi=400,format='png')


# %%
# Testing only
column='Alley'
plt.figure()
house.all.groupby([column,'test']).size().unstack().plot.bar()
plt.savefig('DistributionChart'+str(column)+'.png', bbox_inches='tight' ,dpi=500,transparent=True, format='png')

# %%
