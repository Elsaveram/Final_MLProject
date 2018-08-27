# More EDA
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
# %%

house.all.MoSold

df=house.train()[['LotFrontage','Neighborhood']]
plt.figure(figsize=(20,8))
df.boxplot(by='Neighborhood', column='LotFrontage')
plt.ylabel('Lot Frontage (ft)')
plt.xticks(rotation=75)
plt.savefig('Figures/LotFrontage.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')


df=house.train()[['SalePrice','Neighborhood']]
plt.figure(figsize=(20,8))
df.boxplot(by='Neighborhood', column='SalePrice')
plt.ylabel('Sale Price $')
plt.xticks(rotation=75)
plt.savefig('Figures/Neighborhood.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

df=house.train()[['SalePrice','YrSold']]
plt.figure(figsize=(20,8))
df.boxplot(by='YrSold', column='SalePrice')
# plt.xticks(np.arange(1, 13),['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'])
# plt.title('Sale prices by Months')
plt.savefig('YearSold.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')


# Plotting Sale Price by month
plt.scatter(house.train()['MoSold'],house.train()['SalePrice'])

df=house.train()[['SalePrice','MoSold']]
plt.figure(figsize=(20,8))
df.boxplot(by='MoSold', column='SalePrice')
plt.xticks(np.arange(1, 13),['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'])
# plt.title('Sale prices by Months')
plt.savefig('MonthsSold.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

data2=house.all

def bar_plot(var_name):
    temp = pd.DataFrame(data2.groupby(var_name)['Id'].count()).reset_index()
    plt.bar(list(temp[var_name]),list(temp['Id']),align='center', alpha=0.5)
    plt.title(var_name + ' distribution')
    return plt.show()

[bar_plot(discrete_vars[i]) for i in range(0,len(discrete_vars))]


#


temp = pd.DataFrame(data2.groupby('YrSold')['SalePrice'].count()).reset_index()
temp
temp.loc[temp['YrSold']==2010,'SalePrice']=temp.loc[temp['YrSold']==2010]['SalePrice']*12/6
plt.bar(list(temp['YrSold']),list(temp['SalePrice']),align='center', alpha=0.5,color=[(0.3,0.5,0.4,0.6),(0.3,0.6,0.4,0.6),(0.3,0.7,0.4,0.6),(0.3,0.8,0.4,0.6),(0.3,0.9,0.4,0.6)])
plt.title('Year Sold distribution')
plt.savefig('Figures/YearDistribution.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

my_color_map=[(0.3,0.0,0.4,0.6),
(0.3,0.1,0.4,0.6),(0.3,0.2,0.4,0.6),(0.3,0.3,0.4,0.6),(0.3,0.4,0.4,0.6),
(0.3,0.5,0.4,0.6),(0.3,0.6,0.4,0.6),(0.3,0.7,0.4,0.6),(0.3,0.8,0.4,0.6),(0.3,0.9,0.4,0.6),(0.3,1.0,0.4,0.6),(0.3,1.0,0.4,0.6)]
temp = pd.DataFrame(data2.groupby('MoSold')['SalePrice'].count()).reset_index()
#%%
plt.figure(figsize=(12,5))
plt.bar(list(temp['MoSold']),list(temp['SalePrice']),align='center', alpha=0.5,color=my_color_map)
plt.title('Month Sold distribution')
# locs, labels = plt.xticks()
plt.xticks(np.arange(1, 13),['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'])
plt.savefig('Figures/YearDistribution.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')
#%%


#
# label = temp.SalePrice
# # Text on the top of each barplot
# r4=
# for i in range(len(r4)):
#     plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)
#
