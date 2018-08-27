
## Data loading
# %%
%reload_ext autoreload
%autoreload 2

from house import *
from config import *
del house
house = House('data/train.csv','data/test.csv')
# %%


house.cleanRP()
##EDA:
#All variables:
house.all.shape
house.all.head()
house.all.dtypes

house.test().shape
house.train().shape


#Response varriable:
house.train().SalePrice.describe()
house.log_transform(house.train().SalePrice)
house.corr_matrix(house.train(), 'SalePrice')

#Missing values:
house.missing_stats()
house.cleanRP()
house.all['Exterior1st'].dtypes

house.sg_ordinals()
house.label_encode_engineer()
house.label_df.sample(10)

columns_to_convert = [  ('MSSubClass', 'object'), ('LotArea', 'float64' ), ('OverallQual', 'object'),
                        ('OverallCond', 'object'), ('1stFlrSF', 'float64'), ('2ndFlrSF', 'float64'),
                        ('3SsnPorch', 'float64'), ('EnclosedPorch', 'float64'), ('GarageCars', 'int64'),
                        ('WoodDeckSF', 'float64'), ('ScreenPorch', 'float64'), ('OpenPorchSF', 'float64'),
                        ('MiscVal', 'float64'), ('LowQualFinSF', 'float64'), ('GrLivArea', 'float64'), ]

house.convert_types(HOUSE_CONFIG)



for category in [x for x in house.all.columns if house.all[x].dtype == 'object']:
    print("Category " + category + " has n unique values " + str(house.all[category].nunique() / house.all.shape[0] * 100) + "%" )

house.distribution_charts()

# Understand the Lot Frontage/Area/Config relationship
#house.relation_stats('LotFrontage', 'LotArea', 'LotConfig')

##Feature Engeneneering:
house.engineer_features()

house.sk_random_forest(500)

house.engineer_features(HOUSE_CONFIG)


house.sk_random_forest(1000)
import matplotlib.pyplot as plt
plt.scatter(house.all['LotFrontage'],house.all['SalePrice'])
