
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


house.cleanRP()

house.convert_types(HOUSE_CONFIG)

house.engineer_features(HOUSE_CONFIG)
# split train test

train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)

# RandomForest
randomForest = ensemble.RandomForestRegressor()
randomForest.set_params(random_state=500,n_jobs=-1)
randomForest.fit(X_test, Y_test) # fit
randomForest.score(X_test, Y_test) # accuracy
randomForest.predict(X_test)
mse = mean_squared_error(Y_test, randomForest.predict(X_test))
print("MSE: %.4f" % mse)
rmse = np.sqrt(-cross_val_score(randomForest, X_train, Y_train, scoring="neg_mean_squared_log_error", cv = 5))
np.mean(rmse)
rmse

grid_para_forest = [{
    "n_estimators": [25, 50, 100],
    "min_samples_leaf": range(1,3),
    "min_samples_split": np.linspace(start=2, stop=30, num=2, dtype=int)}]
grid_search_forest = GridSearchCV(estimator=randomForest, param_grid=grid_para_forest, scoring="neg_mean_squared_log_error", cv=5, n_jobs=-1)
%time grid_search_forest.fit(X_train, Y_train)

#y_pred_train = grid_search.best_estimator_.predict(x_train)  #Why best_estimator
y_pred_test = grid_search_forest.best_estimator_.predict(X_test)
mse = mean_squared_error(Y_test, y_pred_test)
print("MSE: %.4f" % mse)
rmse = np.sqrt(-cross_val_score(grid_search_forest, X_train, Y_train, scoring="neg_mean_squared_log_error", cv = 5))
np.mean(rmse)
rmse



















# Boosting
params={'n_estimators': 3000, 'max_depth': 4, 'min_samples_split': 6,
                 'learning_rate': 0.01, 'loss':'ls', 'max_features':18}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, Y_train)
clf.predict(X_test)
importance_list=list(zip(X_train.columns,clf.feature_importances_))
really_important=()
for i in range(0,len(importance_list)):

    if importance_list[i][1]>0.01:
        really_important=really_important+importance_list[i]
importance_list
len(really_important)

important_col=list(really_important[0:47:2])

# error claculations:
mse = mean_squared_error(Y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
rmse = np.sqrt(-cross_val_score(clf, X_train, Y_train, scoring="neg_mean_squared_log_error", cv = 5))
np.mean(rmse)
rmse


#creatinf file for Kaggel
ID=pd.DataFrame(house.Id)['Id']
predict=pd.DataFrame(clf.predict(test),columns=['SalePrice'])
predict['ID']=list(range(1461,2920))
predict=predict[['ID','SalePrice']]
predict.to_csv('predict',index=False)


X_train_new=train[important_col]
X_test_new=test[important_col]
y_train=train['SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split( X_train_new , y_train)
params={'n_estimators': 3000, 'max_depth': 4, 'min_samples_split': 6,
                 'learning_rate': 0.01, 'loss':'ls', 'max_features':18}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, Y_train)
clf.predict(X_test)
mse = mean_squared_error(Y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
rmse = np.sqrt(-cross_val_score(clf, X_train, Y_train, scoring="neg_mean_squared_log_error", cv = 5))
np.mean(rmse)
rmse



train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)





xg=ensemble.GradientBoostingRegressor()
grid_para_xg =[{'n_estimators': [30000], 'max_depth': [4], 'min_samples_split': [6],
                 'learning_rate': [0.0023,0.002,0.0025], 'loss':['ls'], 'max_features':[18]}]

#max depth 4 #min_sample:6
#max_features 18
grid_search_xg = model_selection.GridSearchCV(xg, grid_para_xg, scoring='neg_mean_squared_error', cv=3, return_train_score=True,  n_jobs=-1)
#scoring:mse
grid_search_xg.fit(X_train, Y_train)

grid_search_xg.predict(X_test)
mse = mean_squared_error(Y_test,grid_search_xg.predict(X_test))
print("MSE: %.4f" % mse)
rmse = np.sqrt(-cross_val_score(grid_search_xg, X_train, Y_train, scoring="neg_mean_squared_log_error", cv = 5))
np.mean(rmse)
rmse
