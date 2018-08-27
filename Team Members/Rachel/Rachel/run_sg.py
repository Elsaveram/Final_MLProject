
## Data loading
# %%
%load_ext autoreload
%autoreload 2

from house import *
del house
house = House('data/train.csv','data/test.csv')
# %%



##EDA:

#Response varriable:
house.train().SalePrice.describe()
house.log_transform(house.train().SalePrice)
# house.corr_matrix(house.train(), 'SalePrice')

#Missing values:
#house.missing_stats()
house.clean()

columns_to_convert = [  ('MSSubClass', 'object'), ('LotArea', 'float64' ), ('OverallQual', 'object'),
                        ('OverallCond', 'object'), ('1stFlrSF', 'float64'), ('2ndFlrSF', 'float64'),
                        ('3SsnPorch', 'float64'), ('EnclosedPorch', 'float64'), ('GarageCars', 'int64'),
                        ('WoodDeckSF', 'float64'), ('ScreenPorch', 'float64'), ('OpenPorchSF', 'float64'),
                        ('MiscVal', 'float64'), ('LowQualFinSF', 'float64'), ('GrLivArea', 'float64'), ]

house.convert_types(columns_to_convert)

house.distribution_charts()

# now check skewness and inspect distribution charts again
house.sg_skewness(mut=0)
for var in house.skewed_features:
    house.log_transform(house.train()[var])

##Feature Engeneneering:
house.engineer_features()
# creates ordinals and then label_df with label encoded categories
house.sg_ordinals()
house.label_encode_engineer()
# house.label_df.sample(10)

house.sg_random_forest(500,'dummy')

house.sg_random_forest(500,'label_df')

#### Now log transform the skewed variables and compare the Random Forest:
house.sg_skewness(mut=1) # will perform the log_transform

house.sg_random_forest(500,'dummy')

house.sg_random_forest(500,'label_df')
# house.simple_model()
