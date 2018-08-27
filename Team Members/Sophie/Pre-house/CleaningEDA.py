
# coding: utf-8

# There's a fill dummy function in Pandas

import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv('all/train.csv')

print(df.shape)
df.sample(10)

miss=df.loc[:,df.isnull().sum()!=0]
miss.sample(10)

# Percentage of Missing:
miss.isnull().sum()/df.shape[0]*100
# creates a series object

df.loc[df.LotFrontage.isnull()]
df.loc[df.MSSubClass in [60,70,75]]

df.Fence.value_counts()

print(df.Alley.unique())
df.loc[df.Alley=='Pave'].shape #50 times Gravel, 41 times Paved Do we need Alley at all?


# PoolArea: Pool area in square feet
#
# PoolQC: Pool quality
#
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool

df.PoolQC.unique() #maybe check a trend for Excellent pools?


# FireplaceQu will depend on number of Fireplaces, col right before it
#
# Fireplaces: Number of fireplaces
#
# FireplaceQu: Fireplace quality
#
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace

garage=df[["GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"]]

garage.GarageType.unique()

# **Some useful functions:**
#
# df.dropna(thresh=5)
#
# For imputation:
# df["postTestScore"].fillna(df.groupby("sex")["postTestScore"].transform("mean"), inplace=True)
# df[df['age'].notnull() & df['sex'].notnull()]

# Sunanda's Correlation matrix:
import seaborn as sns
from matplotlib import pyplot as plt
# cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
corr = df.corr()
f, ax=plt.subplots(figsize=(16,12))
sns.heatmap(corr)

def magnify():
    return [dict(selector="th",
                props=[("font-size", "7pt")]),
           dict(selector="td",
                props=[('padding', "0em 0em")]),
           dict(selector="th:hover",
                props=[("font-size", "12pt")]),
           dict(selector="tr:hover td:hover",
                props=[('max-width', '200px'),
                       ('font-size', '12pt')])]

corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '10px', 'font-size': '8pt'}).set_precision(2)
#.set_table_styles(magnify())
#.set_caption("Hover to magify")
