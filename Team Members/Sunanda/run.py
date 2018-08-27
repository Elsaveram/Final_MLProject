
## Data loading
# %%
%reload_ext autoreload
%autoreload 2

from house import *
from config import *
del house
house = House('data/train.csv','data/test.csv')
# %%

#house.log_transform(house.train().SalePrice)
house.cleanRP()
house.all.Alley.unique()
house.all.BsmtCond.unique()
house.convert_types(HOUSE_CONFIG)
house.ordinal_features(HOUSE_CONFIG)
house.one_hot_features()

house.sm_addFeatures()
house.all.columns


#house.test_train_split()
#train = house.dummy_train.drop('SalePrice', axis = 1)
#train2 = train.drop('test',axis = 1)

from sklearn.cross_validation import train_test_split
x=house.train().drop(['SalePrice','test'],axis=1)
y=house.train().SalePrice
x_train, x_test, y_train, y_test = train_test_split(x,y)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.datasets import make_regression

type(x_train)
x_train_columns = x_train.columns
x_train, y_train = make_regression(random_state=0)
x_train_columns
regr = ElasticNet(random_state=0)

regr.fit(x_test, y_test)
len(regr.coef_)

type(x_train)
importance_list = list(zip(x_train_columns, regr.coef_))
importance_list[0][1]
really_important=[]
really_important_beta = []
for variable,beta in importance_list:
    if beta < -1 or beta > 1:
        really_important.append(variable)
        really_important_beta.append(beta)
really_important_list = list(zip(really_important,really_important_beta))
really_important_list
len(importance_list)
len(really_important)
really_important


house.x_train, house.y_train = make_regression(random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(house.x_test, house.y_test)
len(regr.coef_)

print(regr.intercept_)
house.sm_boxcox()
print(regr.predict([[0, 0]]))


###OLD CODE
house.testmethod()
house.train().SalePrice.describe()
house.corr_matrix(house.train(), 'SalePrice')
house.missing_stats()
house.all.head()
columns_to_convert = [  ('MSSubClass', 'object'), ('LotArea', 'float64' ), ('OverallQual', 'object'),
                        ('OverallCond', 'object'), ('1stFlrSF', 'float64'), ('2ndFlrSF', 'float64'),
                        ('3SsnPorch', 'float64'), ('EnclosedPorch', 'float64'), ('GarageCars', 'int64'),
                        ('WoodDeckSF', 'float64'), ('ScreenPorch', 'float64'), ('OpenPorchSF', 'float64'),
                        ('MiscVal', 'float64'), ('LowQualFinSF', 'float64'), ('GrLivArea', 'float64'),
                        ('GarageCars', 'int64')]
house.convert_types(columns_to_convert)
house.sale_price_charts()
for category in [x for x in house.all.columns if house.all[x].dtype == 'object']:
    print("Category " + category + " has n unique values " + str(house.all[category].nunique() / house.all.shape[0] * 100) + "%" )
house.distribution_charts()

#check
from scipy.special import boxcox1p
from scipy.stats import skew
skewness = house.train().select_dtypes(exclude = ["object"]).apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.9]
skewed_features = skewness.index
skewed_features
lam = 0.15
house.train()[skewed_features] = boxcox1p(house.train()[skewed_features], lam)
house.train().Pool
house.test_train_split()

house.sm_boxcox(mut = 1)

house.train().head()

house.all.TotalBath.head()

# Understand the Lot Frontage/Area/Config relationship
#house.relation_stats('LotFrontage', 'LotArea', 'LotConfig')

##Feature Engeneneering:
house.engineer_features(HOUSE_CONFIG)


house.sk_random_forest(1000)
