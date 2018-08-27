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
#%%

house.cleanRP()

plt.scatter(house.all['LotFrontage'],house.all['LotArea'])

plt.scatter(np.log(house.all['LotArea']),house.all['LotFrontage'],c=house.all['plotLotFrontage'])
plt.title('Log LotArea')
plt.xlabel('Log Lot Area')
plt.ylabel('Lot Frontage')
plt.savefig('Figures/LogImputation.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

# %%
house_sg=House('data/train.csv','data/test.csv')

house_sg.cleanRP_SG()

plt.scatter(house_sg.all['LotArea'],house_sg.all['LotFrontage'],c=house_sg.all['plotLotFrontage'])
plt.title('Imputed with Regression on Log Lot Area')
plt.xlabel('Lot Area')
plt.ylabel('Lot Frontage')
plt.savefig('Figures/LogRegressionImputation.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')
