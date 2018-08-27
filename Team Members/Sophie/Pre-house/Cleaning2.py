import pandas as pd
import numpy as np

df=pd.read_csv('all/train.csv')
df.columns
##### For when we want to combine datasets like in the kaggle workflow
# train_df = pd.read_csv('all/train.csv')
# test_df = pd.read_csv('all/test.csv')
#
# # Save the 'Id' column
# train_ID = train_df['Id']
# test_ID = test_df['Id']
#
# # Now drop the 'Id' colum since we can not use it as a feature to train our model.
# train_df.drop("Id", axis = 1, inplace = True)
# test_df.drop("Id", axis = 1, inplace = True)
#
# y_train = train_df['SalePrice']
# X_train = train_df.drop('SalePrice', axis=1)
# X_test = test_df.copy()
#
# del train_df, test_df
#
# df = pd.concat([X_train, X_test], ignore_index=True)
# df.shape

one_hot_df = pd.get_dummies(df, drop_first=True, dummy_na=True) # new column will be DummyNA :/
n,p=one_hot_df.shape
n/p #since this is <20:1 ratio, we should one hot encode this dataset
del one_hot_df,n,p

# let's give the ordinal columns numbers
ord_cols = ['ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'KitchenQual',
           'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1}

for col in ord_cols:
    df[col] = df[col].map(lambda x: ord_dic.get(x, 0))
df.shape
n,p=df.shape
n/p
# df.loc[df.Heating.isnull()]
# %% Imputing LotFrontage

dfF=df[[ 'LotFrontage', 'LotArea', 'LotConfig', 'LotShape']]
dfF=pd.get_dummies(dfF,drop_first=True,dummy_na=True)
train=dfF.loc[dfF.LotFrontage.isnull()==False]
test=dfF.loc[dfF.LotFrontage.isnull()==True]
test=test.drop('LotFrontage',axis=1)

train.sample(10)
n,p=df.shape
n/p # good shape for dummification
# %%
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
X=train.drop('LotFrontage',axis=1)
X=X.astype(float)
Y=train.LotFrontage.astype(float)
print(X.shape,Y.shape)
ols.fit(X, Y)
y_predict = ols.predict(test)

print("beta_1, beta_2: " + str(np.round(ols.coef_, 3)))
print("beta_0: " + str(np.round(ols.intercept_, 3)))
# not sure that this RSS is correct
print("RSS: %.2f" % np.sum((y_predict - Y.mean()) ** 2))
print("R^2: %.5f" % ols.score(X, Y))

# %% Separating out my Separating

#dataframeSophie

dfS=df[['SalePrice','Heating','HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', \
'2ndFlrSF', \
'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', \
'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', \
'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu']]

dfS[['GrLivArea','1stFlrSF']].corr()
dfS.GrLivArea.value_counts()
dfS[['GrLivArea','2ndFlrSF']].corr()

dfS.BsmtHalfBath.value_counts()
# Wonder if the Bsmt bathrooms parameters covary with BsmtCond or BsmtFinType1

# %% Heating Variables
from LabelClass import LabelCountEncoder
lce = LabelCountEncoder()
dfS['Heating'] = lce.fit_transform(dfS['Heating'])

dfS[['HeatingQC','Heating']].corr()
df.HeatingQC.value_counts()
df.Heating.value_counts()

# %%

df.loc[df.Functional=='Maj2']
dfS.loc[dfS.Functional==3]
dfS['Kitchen*Quality']=dfS.KitchenAbvGr*dfS.KitchenQual

functional_dic = {'Typ':8, 'Min1':7,'Min2': 6,'Mod':5, 'Maj1':4,'Maj2':3,'Sev':2,'Sal':1}
dfS['Functional'] = dfS.loc[:,'Functional'].map(lambda x: functional_dic.get(x, 0))

# %% Missingness
dfS.dtypes
dfS=dfS.dropna(subset=['Electrical'])
dfS.BsmtFullBath=dfS.BsmtFullBath.fillna(0.0)
dfS.BsmtHalfBath=dfS.BsmtHalfBath.fillna(0.0)
dfS.Functional=dfS.Functional.fillna('Typ')

miss=dfS.loc[:,dfS.isnull().sum()!=0]
dfS.Functional.unique()
dfS.loc[dfS.Functional.isnull()]
miss.columns

# %% Feature engineering:
dfS['Kitchen*Quality']=dfS.KitchenAbvGr*dfS.KitchenQual
dfS['Fireplace*Quality']=dfS['FireplaceQu']*dfS['Fireplaces']
# dfS=dfS.drop(['Fireplaces','FireplaceQu','KitchenQual','KitchenAbvGr'],axis=1)
# %% Histograms for normalcy:
dfS.columns
dfS.BedroomAbvGr.unique()
plt.hist(dfS['BedroomAbvGr'])

# %% Scatterplots & Boxplots
import matplotlib.pyplot as plt

def sctplot(x,i):
    plt.scatter(x,dfS['SalePrice'])
    plt.title('SalePrice vs ' + str(num_colnames[i]))
    return plt.show()

num_colnames=dfS.describe().columns[1:-1]
[sctplot(dfS[num_colnames[i]],i) for i in range(0,len(num_colnames))]

dfS.LowQualFinSF.value_counts()
1434/1460

plt.scatter(dfS['1stFlSF'],dfS['SalePrice'])
plt.title('SalePrice vs 1stFlrSF')
plt.show()

plt.scatter(dfS['FireplaceQu']*dfS['Fireplaces'],dfS['SalePrice'])
plt.title('SalePrice vs Fireplaces*Quality')
plt.show()

plt.scatter(dfS['KitchenQual']*dfS['KitchenAbvGr'],dfS['SalePrice'])
plt.title('SalePrice vs Kitchen*Quality')
plt.show()
