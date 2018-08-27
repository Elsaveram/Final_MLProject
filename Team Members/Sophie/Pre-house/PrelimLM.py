import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import linear_model
ols = linear_model.LinearRegression()
# Also valid Python syntax:
# from sklearn.linear_model import LinearRegression
# ols = LinearRegression()
df=pd.read_csv('all/train.csv')
df.dtypes
df_tmp=df[['LotArea','SalePrice']]
df_tmp=df_tmp[df_tmp.LotArea<30000]
df_tmp=df_tmp.dropna()
X = df_tmp.LotArea
Y = df_tmp.SalePrice
ols.fit(X.reshape(-1,1), Y)
print("beta_1: %.3f" %ols.coef_)
print("beta_0: %.3f" %ols.intercept_)

# X = X.reshape(-1,1)

# %% Plot figure Run cell = option+shift+return
plt.figure(figsize=(9, 6))
# plt.plot(X, beta_0 + beta_1*X, c='b', lw=1.5, label='True relation')
plt.plot(X, ols.predict(X.reshape(-1,1)), c='r', lw=1.5, label='Predicted relation')
plt.scatter(X, Y, color='green')
plt.xlabel('LotArea')
plt.ylabel('Price')
plt.legend(loc=2)
plt.show()

# %%
