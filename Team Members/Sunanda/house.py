import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#Sophie's additions
from LabelClass import LabelCountEncoder
from scipy.stats import skew

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from scipy.special import boxcox1p

from statsmodels.formula.api import ols
import statsmodels.api as sm
from  statsmodels.genmod import generalized_linear_model

#import missingno as msno

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

# A class to hold our housing data
class House():
    def __init__(self, train_data_file, test_data_file):
        train = pd.read_csv(train_data_file)
        test = pd.read_csv(test_data_file)
        self.all = pd.concat([train,test], ignore_index=True)
        self.all['test'] = self.all.SalePrice.isnull()
        self.all.drop('Id', axis=1, inplace=True)

    def train(self):
        return(self.all[~self.all['test']])

    def test(self):
        return(self.all[self.all['test']])

    def log_transform(self, variable):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.distplot(variable, bins=50)
        plt.title('Original')
        plt.subplot(1,2,2)
        sns.distplot(np.log1p(variable), bins=50)
        plt.title('Log transformed')
        plt.tight_layout()

    def corr_matrix(self, data, column_estimate, k=10, cols_pair=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']):
        corr_matrix = data.corr()
        sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='coolwarm')
        plt.figure()

        cols = corr_matrix.nlargest(k, column_estimate)[column_estimate].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=1.25)
        f, ax = plt.subplots(figsize=(12, 9))
        hm = sns.heatmap(cm, cbar=True, cmap='coolwarm', annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        plt.figure()

        sns.set()
        sns.pairplot(data[cols_pair], size = 2.5)
        plt.show()

    def missing_stats(self):
        # Basic Stats
        self.all.info()

        # Heatmap
        sns.heatmap(self.all.isnull(), cbar=False)
        col_missing=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        col_missing.remove('SalePrice')
        print(col_missing)
        msno.heatmap(self.all)
        plt.figure()
        msno.heatmap(self.all[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']])
        plt.figure()
        msno.heatmap(self.all[['GarageCond', 'GarageFinish', 'GarageFinish', 'GarageQual','GarageType', 'GarageYrBlt']])
        plt.figure()
        msno.dendrogram(self.all)
        plt.figure()

        # Bar chart
        if len(col_missing) != 0:
            plt.figure(figsize=(12,6))
            np.sum(self.all[col_missing].isnull()).plot.bar(color='b')

            # Table
            print(pd.DataFrame(np.sum(self.all[col_missing].isnull())))
            print(np.sum(self.all[col_missing].isnull())*100/self.all[col_missing].shape[0])


    def distribution_charts(self):
        for column in self.all.columns:
            if self.all[column].dtype in ['object', 'int64']:
                plt.figure()
                self.all.groupby([column,'test']).size().unstack().plot.bar()

            elif self.all[column].dtype in ['float64']:
                plt.figure(figsize=(10,5))
                sns.distplot(self.all[column][self.all[column]>0])
                plt.title(column)

    def boxplots(self):
        for column in self.all.columns:
            if self.all[column].dtype in ['object', 'int64']:
                plt.figure()
                self.all.groupby([column,'test']).size().unstack().boxplot()

    def relation_stats(self, x, y, z):
        # x vs y scatter
        plt.figure()
        self.all.plot.scatter(x, y)
        print(self.all[[x, y]].corr(method='pearson'))

        # z vs x box
        df_config = self.all[[z, x]]
        df_config.boxplot(by=z, column=x)
        mod_2 = ols( x + ' ~ ' + z, data=df_config).fit()

        aov_table = sm.stats.anova_lm(mod_2, typ=2)
        print(aov_table)

        #LotFrontage vs LotShape #significant
        df_frontage = self.all[['LotShape', 'LotFrontage']]
        df_frontage.boxplot(by='LotShape', column='LotFrontage')

        mod = ols('LotFrontage ~ LotShape', data=df_frontage).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        print(aov_table)


    def clean(self):
        columns_with_missing_data=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        columns_with_missing_data.remove('SalePrice')

        for column in columns_with_missing_data:
            col_data = self.all[column]
            print( 'Cleaning ' + str(np.sum(col_data.isnull())) + ' data entries for column: ' + column )

            if column == 'Electrical':
                # TBD: Impute based on a distribution
                self.all[column] = [ 'SBrkr' if pd.isnull(x) else x for x in self.all[column]]
            elif column == 'LotFrontage':
                self.all[column].fillna(self.all[column].mean(),inplace=True)
            elif column == 'GarageYrBlt':
                # TBD: One house has a detached garage that could be caclulatd based on the year of construction.
                self.all[column] = [ 'NA' if pd.isnull(x) else x for x in self.all[column]]
            elif column in ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','MasVnrArea']:
                self.all[column] = [ 0 if pd.isnull(x) else x for x in self.all[column]]
            elif col_data.dtype == 'object':
                self.all[column] = [ "None" if pd.isnull(x) else x for x in self.all[column]]
            else:
                print( 'Uh oh!!! No cleaning strategy for:' + column )

    def testmethod(self):
        print("this is a test")
        917-494-5827


    def sm_addFeatures(self):
        self.all['OverallGrade'] = self.all['OverallQual'] * self.all['OverallCond']
        self.all['GarageGrade'] = self.all['GarageQual'] * self.all['GarageCond']
        self.all['ExterGrade'] = self.all['ExterQual'] * self.all['ExterCond']
        self.all['KitchenScore'] = self.all['KitchenAbvGr'] * self.all['KitchenQual']
        self.all['FireplaceScore'] = self.all['Fireplaces'] * self.all['FireplaceQu']
        self.all['GarageScore'] = self.all['GarageArea'] * self.all['GarageQual']
        self.all['PoolScore'] = self.all['PoolArea'] * self.all['PoolQC']
        self.all['TotalBath'] = self.all['BsmtFullBath'] + (0.5 * self.all['BsmtHalfBath']) + self.all['FullBath'] + (0.5 * self.all['HalfBath'])
        self.all['AllSF'] = self.all['GrLivArea'] + self.all['TotalBsmtSF'] + self.all['1stFlrSF'] + self.all['2ndFlrSF']
        self.all['AllPorchSF'] = self.all['3SsnPorch'] + self.all['OpenPorchSF'] + self.all['EnclosedPorch'] + self.all['ScreenPorch']

        self.all.drop(['OverallQual','OverallCond', 'GarageQual', 'GarageCond', 'ExterQual',
        'ExterCond','KitchenAbvGr', 'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageArea', 'GarageQual',
        'PoolArea', 'PoolQC', 'ExterQual', 'ExterCond', 'PoolArea', 'PoolQC', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'OpenPorchSF',
        'EnclosedPorch', 'ScreenPorch'],axis = 1,inplace=True)

    def cleanRP(self):
        NoneOrZero=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','BsmtFinSF1','BsmtFinSF2','Alley',
               'Fence','GarageType','GarageQual',
               'GarageCond','GarageFinish','GarageCars',
                'GarageArea','MasVnrArea','MasVnrType','MiscFeature','PoolQC',
                'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']
        mode=['Electrical','Exterior1st','Exterior2nd','FireplaceQu','Functional','KitchenQual','MSZoning','SaleType','Utilities']
        mean=['TotalBsmtSF']
        columns_with_missing_data=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        columns_with_missing_data.remove('SalePrice')
        for column in columns_with_missing_data:
            col_data = self.all[column]
            print( 'Cleaning ' + str(np.sum(col_data.isnull())) + ' data entries for column: ' + column )

        #log transformation for missing LotFrontage
            if  column=='LotFrontage':
                y1=np.log(self.all['LotArea'])
                index=self.all[self.all['LotFrontage'].isnull()].index
                self.all.loc[self.all['LotFrontage'].isnull(),'LotFrontage'] = y1.loc[index]
            #imputing the value of YearBuiltto the GarageYrBlt.
            elif  column=='GarageYrBlt':
                missing_grage_yr=self.all[self.all['GarageYrBlt'].isnull()].index
                self.all.loc[self.all['GarageYrBlt'].isnull(),'GarageYrBlt'] = self.all['YearBuilt'].loc[missing_grage_yr]

            elif column in mode:
                self.all[column] = [self.all[column].mode()[0] if pd.isnull(x) else x for x in self.all[column]]
            elif column in mean:
                self.all[column].fillna(self.all[column].mean(),inplace=True)
            elif column in NoneOrZero:
                if col_data.dtype == 'object':
                    no_string = 'None'
                    self.all[column] = [ no_string if pd.isnull(x) else x for x in self.all[column]]
                else:
                    self.all[column] = [ 0 if pd.isnull(x) else x for x in self.all[column]]
            else:
                print( 'Uh oh!!! No cleaning strategy for:' + column )


    def convert_types(self, columns_to_convert):
        for column, type in columns_to_convert:
            print("assigning " + column + " as type " + type)
            self.all[column] = self.all[column].astype(type)


    def engineer_features(self, house_config):
        # General Dummification
        categorical_columns = [x for x in self.train().columns if self.train()[x].dtype == 'object' ]
        non_categorical_columns = [x for x in self.train().columns if self.train()[x].dtype != 'object' ]

        # TBD: do something with ordinals!!!!!
        for column in categorical_columns:
            for member_name, member_dict in house_config[column]['members'].items():
                if member_dict['ordinal'] != 0:
                    print( "Replacing " + member_name + " with " + str(member_dict['ordinal']) + " in column " + column)
                    self.all[column].replace(member_name, member_dict['ordinal'], inplace=True)

            #print( "Column " + column + " now has these unique values " + ' '.join(self.all[column].unique()))

        use_columns = non_categorical_columns + categorical_columns
        self.dummy_train = pd.get_dummies(self.train()[use_columns], drop_first=True, dummy_na=True)
    def convert_types(self, house_config):
        for house_variable_name, house_variable_value in house_config.items():
            if len(house_variable_value['dtype']) != 0:
                print("assigning " + house_variable_name + " as type " + house_variable_value['dtype'])
                self.all[house_variable_name] = self.all[house_variable_name].astype(house_variable_value['dtype'])


    def ordinal_features(self, house_config):
        self.categorical_columns = [x for x in self.all.columns if self.all[x].dtype == 'object' ]
        self.non_categorical_columns = [x for x in self.all.columns if self.all[x].dtype != 'object' ]

        for column in self.categorical_columns:
           for member_name, member_dict in house_config[column]['members'].items():
               if member_dict['ordinal'] != 0:
                   print( "Replacing " + member_name + " with " + str(member_dict['ordinal']) + " in column " + column)
                   self.all[column].replace(member_name, member_dict['ordinal'], inplace=True)

    def one_hot_features(self):
       self.use_columns = self.non_categorical_columns + self.categorical_columns
       self.all = pd.get_dummies(self.all[self.use_columns], drop_first=True, dummy_na=True)


    def sg_test_train_split(self,data_type):
        if data_type=="label_df":
            x=self.label_df
        elif data_type=='dummy':
            x=self.dummy_train
        y=self.train().SalePrice
        try:
            self.x_train
        except:
            print('DOING SPLITS!!!!')
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y)

    def sg_skewness(self,mut=0): # mut=0 will not log transform, mut =1 will
        skewness = self.train().select_dtypes(exclude = ["object"]).apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        print(str(skewness.shape[0]) + " skewed numerical features to log transform")
        skewed_features = skewness.index
        if mut==1:
            self.train()[skewed_features] = np.log1p(self.train()[skewed_features])
        self.skewed_features=skewness.index
        print(skewed_features)


    def sg_random_forest(self,num_est=500,data_type='dummy'):
        self.sg_test_train_split(data_type=data_type)

        model_rf = RandomForestRegressor(n_estimators=num_est, n_jobs=-1)
        model_rf.fit(self.x_train, self.y_train)
        rf_pred = model_rf.predict(self.x_test)

        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, rf_pred, s=20)
        plt.title('Predicted vs. Actual')
        plt.xlabel('Actual Sale Price')
        plt.ylabel('Predicted Sale Price')

        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)])
        plt.tight_layout()

        model_rf.fit(self.x_train, self.y_train)
        rf_pred_log = model_rf.predict(self.x_test)

        print(self.rmse_cv(model_rf, self.x_train, self.y_train))

    def label_encode_engineer(self):
        # must be called AFTER sg_ordinals
        lce = LabelCountEncoder()
        self.label_df = self.ord_df.copy()

        for c in self.train().columns:
            if self.label_df[c].dtype == 'object':
                lce = LabelCountEncoder()
                self.label_df[c] = lce.fit_transform(self.label_df[c])

    def sale_price_charts(self):
        for i, column in enumerate(self.all.columns):
            plt.figure(i)
            if column == 'SalePrice':
                pass
            elif self.all[column].dtype == 'float64':
                data = pd.concat([self.all['SalePrice'], self.all[column]], axis=1)
                data.plot.scatter(x=column, y='SalePrice', ylim=(0,800000))
            else:
                var = column
                data = pd.concat([self.all['SalePrice'], self.all[var]], axis=1)
                f, ax = plt.subplots(figsize=(16, 8))
                fig = sns.boxplot(x=var, y="SalePrice", data=data)
                fig.axis(ymin=0, ymax=800000)

    def rmse_cv(self,model, x, y, k=5):
        rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_log_error", cv = k))
        return(np.mean(rmse))


    def statsmodel_linear_regression(self,y=['SalePrice'], X=['GrLivArea']):
        x = sm.add_constant(self.train()[X])
        y = self.train()[y]
        model = sm.OLS(y,x)
        results = model.fit()
        print(results.summary())


    def test_train_split(self):
        x=x=self.dummy_train.drop('SalePrice', axis = 1)
        y=self.train().SalePrice
        try:
            self.x_train
        except:
            print('DOING SPLITS!!!!')
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y)


    def sk_random_forest(self,num_est=500):
        self.test_train_split()

        model_rf = RandomForestRegressor(n_estimators=num_est, n_jobs=-1)
        model_rf.fit(self.x_train, self.y_train)
        rf_pred = model_rf.predict(self.x_test)

        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, rf_pred, s=20)
        plt.title('Predicted vs. Actual')
        plt.xlabel('Actual Sale Price')
        plt.ylabel('Predicted Sale Price')

        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)])
        plt.tight_layout()

        model_rf.fit(self.x_train, self.y_train)
        rf_pred_log = model_rf.predict(self.x_test)

        print(self.rmse_cv(model_rf, self.x_train, self.y_train))
