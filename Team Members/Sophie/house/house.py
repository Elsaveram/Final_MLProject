import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#Sophie's additions
from LabelClass import LabelCountEncoder
from scipy.stats import skew
from sklearn import linear_model

from statsmodels.formula.api import ols
import statsmodels.api as sm
from  statsmodels.genmod import generalized_linear_model

import missingno as msno

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
        self.results_dict={}
    def train(self):
        return(self.all[~self.all['test']])

    def test(self):
        return(self.all[self.all['test']])

    def sg_split(self):
        print(self.shape)

    def log_transform(self, variable,feat):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        sns.distplot(variable, bins=50)
        plt.title('Original')
        plt.subplot(1,2,2)
        sns.distplot(np.log1p(variable), bins=50)
        plt.title('Log transformed')
        plt.tight_layout()
        plt.savefig('Figures/Skewness/'+str(feat)+'.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

    def corr_matrix(self, data, column_estimate, k=10, cols_pair=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
     'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']):
        corr_matrix = data.corr()
        sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='coolwarm')
        plt.figure()

        cols = corr_matrix.nlargest(k, column_estimate)[column_estimate].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=1.25)
        f, ax = plt.subplots(figsize=(12, 9))
        hm = sns.heatmap(cm, cbar=True, cmap='coolwarm', annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
         yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        plt.figure()

        sns.set()
        sns.pairplot(data[cols_pair], size = 2.5)
        plt.show()

    def missing_stats(self):
        # Basic Stats
        self.all.info()

        # Heatmap
        heat=sns.heatmap(self.all.drop(['SalePrice'],axis=1).isnull(), cbar=False)
        fig=heat.get_figure()
        fig.savefig('Figures/Missingness/missingness.png',transparent=True, dpi=400,bbox_inches='tight' ,format='png')#
        col_missing=[name for name in self.all.columns if np.sum(self.all[name].isnull()) !=0]
        col_missing.remove('SalePrice')
        print(col_missing)

        heat=sns.heatmap(self.all[col_missing],cbar=False)

        heat=msno.heatmap(self.all)
        # plt.figure()
        fig=heat.get_figure()
        fig.savefig('Figures/Missingness/NullityCorrelation.png', transparent=True, bbox_inches='tight' ,dpi=400,format='png')
        heat=msno.heatmap(self.all[['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']])
        # plt.figure()
        fig=heat.get_figure()
        fig.savefig('Figures/Missingness/HeatMapBasement.png',  bbox_inches='tight' ,dpi=400,transparent=True, format='png')
        heat=msno.heatmap(self.all[['GarageCond', 'GarageFinish', 'GarageFinish', 'GarageQual','GarageType', 'GarageYrBlt']])
        # plt.figure()
        fig=heat.get_figure()
        fig.savefig('Figures/Missingness/HeatMapGarage.png',bbox_inches='tight' ,dpi=400,transparent=True, format='png')
        heat=msno.dendrogram(self.all)
        fig=heat.get_figure()
        fig.savefig('Figures/Missingness/Dendrogram.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

        # Bar chart
        if len(col_missing) != 0:
            # IF WE USE THIS WE SHOULD NORMALIZE IT TO PERCENTAGE?
            plt.figure(figsize=(12,6))
            np.sum(self.all[col_missing].isnull()).plot.bar(color='b')
            # plt.savefig('Figures/Missingness/'+str(column)+'.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')
            # Table
            print(pd.DataFrame(np.sum(self.all[col_missing].isnull())))
            print(np.sum(self.all[col_missing].isnull())*100/self.all[col_missing].shape[0])


    def distribution_charts(self):
        for column in self.all.columns:
            if self.all[column].dtype in ['object', 'int64']:
                plt.figure()
                self.all.groupby([column,'test']).size().unstack().plot.bar(color=('c','m'))
                plt.savefig('Figures/DistributionCharts/'+str(column)+'.png',bbox_inches='tight' ,dpi=400,transparent=True,  format='png')

            elif self.all[column].dtype in ['float64']:
                plt.figure(figsize=(10,5))
                sns.distplot(self.all[column][self.all[column]>0],color=('c','m'))
                plt.title(column)
                plt.savefig('Figures/DistributionCharts/'+str(column)+'.png', bbox_inches='tight' ,dpi=400,transparent=True, format='png')

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

    def cleanRP_SG(self):
        from math import exp
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
            #print( 'Cleaning ' + str(np.sum(col_data.isnull())) + ' data entries for column: ' + column )
        #log transformation for missing LotFrontage
            if  column=='LotFrontage':
                ols=linear_model.LinearRegression()
                my_ind=self.all[self.all['LotFrontage'].isnull()].index
                # self.all['plotLotFrontage']=self.all['LotFrontage'].isnull()
                y_train=self.all['LotFrontage'].loc[self.all['LotFrontage'].isnull()==False].values
                x_train=self.all['LotArea'].loc[self.all['LotFrontage'].isnull()==False].values
                x_train=np.log(x_train)
                ols.fit(x_train.reshape(-1,1), y_train)   #### What happen if we remove the 'reshape' method?
                for i in my_ind:
                    print(np.log(self.all['LotArea'].loc[i])*ols.coef_)
                    self.all.loc[i,'LotFrontage']=((np.log(self.all.loc[i,'LotArea'])*ols.coef_)+ols.intercept_)[0]

            #imputing the value of YearBuiltto the GarageYrBlt.
            elif  column=='GarageYrBlt':
                missing_grage_yr=self.all[self.all['GarageYrBlt'].isnull()].index
                self.all.loc[self.all['GarageYrBlt'].isnull(),'GarageYrBlt'] = self.all['YearBuilt'].loc[missing_grage_yr]

            elif column in mode:
                # in case of function messing up - remove [0]
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
            #print( 'Cleaning ' + str(np.sum(col_data.isnull())) + ' data entries for column: ' + column )
            #log transformation for missing LotFrontage
            if  column=='LotFrontage':
                y1=np.log(self.all['LotArea'])
                self.all['plotLotFrontage']=self.all['LotFrontage'].isnull()
                index=self.all[self.all['LotFrontage'].isnull()].index
                self.all.loc[self.all['LotFrontage'].isnull(),'LotFrontage'] = y1.loc[index]
            #imputing the value of YearBuiltto the GarageYrBlt.
            elif  column=='GarageYrBlt':
                missing_grage_yr=self.all[self.all['GarageYrBlt'].isnull()].index
                self.all.loc[self.all['GarageYrBlt'].isnull(),'GarageYrBlt'] = self.all['YearBuilt'].loc[missing_grage_yr]

            elif column in mode:
                # in case of function messing up - remove [0]
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

    # def convert_types(self, columns_to_convert):
    #     for column, type in columns_to_convert:
    #         print("assigning " + column + " as type " + type)
    #         self.all[column] = self.all[column].astype(type)

    def convert_types(self, house_config):
        for house_variable_name, house_variable_value in house_config.items():
            if len(house_variable_value['dtype']) != 0:
                print("assigning " + house_variable_name + " as type " + house_variable_value['dtype'])
                self.all[house_variable_name] = self.all[house_variable_name].astype(house_variable_value['dtype'])

    def engineer_features(self, house_config):
        # General Dummification
        categorical_columns = [x for x in self.all.columns if self.all[x].dtype == 'object' ]
        non_categorical_columns = [x for x in self.all.columns if self.all[x].dtype != 'object' ]

        # TBD: do something with ordinals!!!!!
        for column in categorical_columns:
            for member_name, member_dict in house_config[column]['members'].items():
                if member_dict['ordinal'] != 0:
                    print( "Replacing " + member_name + " with " + str(member_dict['ordinal']) + " in column " + column)
                    self.all[column].replace(member_name, member_dict['ordinal'], inplace=True)

            #print( "Column " + column + " now has these unique values " + ' '.join(self.all[column].unique()))

        use_columns = non_categorical_columns + non_categorical_columns
        self.dummy_train = pd.get_dummies(self.all[use_columns], drop_first=True, dummy_na=True)

    def sg_ordinals(self):
        # general ordinal columns
        ord_cols = ['ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'KitchenQual',
                   'FireplaceQu']
        ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1}
        for col in ord_cols:
            try:
                self.all[col] = self.all[col].apply(lambda x: ord_dic.get(x, 0))
            except:
                pass
        #Different Ordinal columns
        GarageQual_dic = {'Ex': 1, 'Gd': 2, 'TA': 3,'Fa': 4, 'Po': 5,'None':6}
        functional_dic = {'Typ':8, 'Min1':7,'Min2': 6,'Mod':5, 'Maj1':4,'Maj2':3,'Sev':2,'Sal':1}
        GarageFinish_dic = {'Fin': 1, 'RFn': 2, 'Unf': 3, 'None':4}
        GarageCond_dic = {'Ex': 1, 'Gd': 2, 'TA': 3,'Fa': 4, 'Po': 5,'None':6}
        PoolQC_dic = {'Ex': 1, 'Gd': 2, 'TA': 3,'Fa': 4, 'Na': 5, 'None':5}
        Utilities_dic = {'AllPub':1,'NoSewr':2,'NoSeWa':3,'ELO':4}
        try:
            self.all['Utilities'] = self.all['Utilities'].apply(lambda x: Utilities_dic.get(x,0))
            self.all['GarageFinish'] = self.all['GarageFinish'].apply(lambda x: GarageFinish_dic.get(x, 0))
            self.all['Functional'] = self.all['Functional'].apply(lambda x: functional_dic.get(x, 0))
            self.all['GarageQual'] = self.all['GarageQual'].apply(lambda x: GarageQual_dic.get(x, 0))
            self.all['GarageCond'] = self.all['GarageCond'].apply(lambda x: GarageCond_dic.get(x, 0))
            self.all['PoolQC'] = self.all['PoolQC'].apply(lambda x: PoolQC_dic.get(x, 0))
        except:
            pass

    def sm_addFeatures_corr(self):
        # self.all['FireplaceScore'] = self.all['Fireplaces'] * self.all['FireplaceQu']
        # self.all['TotalBath'] = self.all['BsmtFullBath'] + (0.5 * self.all['BsmtHalfBath']) + self.all['FullBath'] + (0.5 * self.all['HalfBath'])
        self.all['AllSF'] = self.all['GrLivArea'] + self.all['TotalBsmtSF'] + self.all['1stFlrSF'] + self.all['2ndFlrSF']

        self.all.drop(['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'],axis = 1,inplace=True)
        # self.all.drop(['Fireplaces', 'FireplaceQu', 'BsmtFullBath', 'BsmtHalfBath',
        # 'FullBath', 'HalfBath', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'],axis = 1,inplace=True)

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


    def sg_skewness(self,mut=0): # mut=0 will not log transform, mut =1 will
    # inspects training data but computes log transform on all the data
        skewness = self.train().drop('SalePrice',axis=1).select_dtypes(exclude = ["object"]).apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        print(str(skewness.shape[0]) + " skewed numerical features to log transform")
        skewed_features = skewness.index
        if mut==1:
            self.all[skewed_features] = np.log1p(self.all[skewed_features])
        self.skewed_features=skewness.index
        print(skewed_features)

    def label_encode_engineer(self):
        lce = LabelCountEncoder()
        for c in self.all.columns:
            if self.all[c].dtype == 'object':
                lce = LabelCountEncoder()
                self.all[c] = lce.fit_transform(self.all[c])

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

    def statsmodel_linear_regression(self,y=['SalePrice'], X=['GrLivArea']):
        x = sm.add_constant(self.all[X])
        y = self.all[y]
        model = sm.OLS(y,x)
        results = model.fit()
        print(results.summary())


    def test_train_split(self):
        x=self.train().drop(['SalePrice','test'],axis=1)
        y=self.train().SalePrice
        try:
            self.x_train
        except:
            print('DOING SPLITS!!!!')
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y)

### MODELS ###

    def sk_random_forest(self,num_est=500):
        self.test_train_split()

        model_rf = RandomForestRegressor(n_estimators=num_est, n_jobs=-1)
        model_rf.fit(self.x_train, self.y_train)
        rf_pred = model_rf.predict(self.x_test)

        self.plot_results(rf_pred)

        model_rf.fit(self.x_train, self.y_train)
        rf_pred_log = model_rf.predict(self.x_test)

        print('RMSLE from Kaggle: '+str(self.rmsle(y_pred=rf_pred,y_test=self.y_test)))
        print('RMSE from Elsa: ' + str(self.rmse_cv(model_rf, self.x_train, self.y_train)))
        self.model_prediction=rf_pred

    def sg_simpleLM(self):
        self.test_train_split()

        x = np.asarray(self.x_train)
        # x = sm.add_constant(x)
        ols = linear_model.LinearRegression()
        # ols = sm.OLS(np.asarray(self.y_train),)
        model = ols.fit(x,np.asarray(self.y_train))#.reshape(-1,1)

        sm_model_pred=model.predict(self.x_test)
        self.plot_results(sm_model_pred)
        print('RMSLE from Kaggle: '+str(self.rmsle(y_pred=sm_model_pred,y_test=self.y_test)))
        print(self.rmse_cv(model, self.x_train, self.y_train))
        self.results_dict['simpleLM']=self.rmse_cv(model, self.x_train, self.y_train)

    def sg_statsmodels(self):
        """gives a good summary of coefficients"""
        self.test_train_split()

        x = np.asarray(self.x_train)
        x = sm.add_constant(x)

        model = sm.OLS(self.y_train, x)
        results = model.fit()
        print(results.summary())

    def elastic_search(self):
        """performs an elastic search.
        Does NOT work with CV for some reason
        """
        self.test_train_split()
        #
        alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]
        l1_ratio=[.01, .1, .5, .9, .99]
        max_iter=5000

        ENSTest = linear_model.ElasticNetCV(alphas, l1_ratio, max_iter)
        ENSTest.fit(self.x_train, self.y_train)
        elast_pred = ENSTest.predict(self.x_test)

        self.plot_results(elast_pred)
        print('RMSLE from Kaggle: '+str(self.rmsle(y_pred=elast_pred,y_test=self.y_test)))
        print(self.rmse_cv(ENSTest, self.x_train, self.y_train))
        self.results_dict['elastic_search']=self.rmse_cv(ENSTest, self.x_train, self.y_train)

### HELPER FUNCTIONS ###
    def save_results(self,model_name,rmse):
        self.results_dict[model_name]=rmse
    def save_kaggle(self,prediction):
        id=range(len(prediction))
        results=pd.DataFrame(id,prediction)
        results.to_csv('Results.csv')

    def rmse_cv(self,model, x, y, k=5):
        rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_log_error", cv = k))
        return(np.mean(rmse))

    def rmsle(self,y_pred, y_test) :
        assert len(y_test) == len(y_pred)
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))

    def plot_results(self,prediction):
        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, prediction, s=20)
        plt.title('Predicted vs. Actual')
        plt.xlabel('Actual Sale Price')
        plt.ylabel('Predicted Sale Price')
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)])
        plt.tight_layout()
