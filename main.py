import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
matplotlib.rcParams['figure.figsize'] = (16,20)
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("train.csv")
train.head()
train.shape
train.info()
cols = train.columns[:81] # I am taking all the 81 features
colours = ['gray','purple'] # Green for non - null values, red for null
sns.heatmap(train[cols].isnull(),cmap = sns.color_palette(colours))
for col in train.columns:
    value_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col,round(value_missing*100)))
    cols_to_drop = ['FireplaceQu', 'PoolQC', 'MiscFeature', 'Fence']
    train_new = train.drop(cols_to_drop, axis=1)
    train_new.shape
    data_numeric = train_new.select_dtypes(include=[np.number])
    numeric_cols = data_numeric.columns.values

    for col in numeric_cols:
        missing = train_new[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            print('imputing missing values for : {}'.format(col))
            train_new['{}_ismissing'.format(col)] = missing
            med = train_new[col].median()
            train_new[col] = train_new[col].fillna(med)
            data_non_numeric = train_new.select_dtypes(exclude=[np.number])
            non_numeric_cols = data_non_numeric.columns.values

            for col in non_numeric_cols:
                missing = train_new[col].isnull()
                num_missing = np.sum(missing)

                if num_missing > 0:
                    print('imputing missing values for : {}'.format(col))
                    train_new['{}_ismissing'.format(col)] = missing

                    top = train_new[col].describe()['top']
                    train_new[col] = train_new[col].fillna(top)
                    train_new.isnull().sum()
                    train_new.describe()
                    print(train_new['SalePrice'].describe())
                    plt.figure(figsize=(9, 8))
                    sns.distplot(train_new['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
                    train_new_num = train_new.select_dtypes(include=['float64', 'int64'])
                    train_new_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
                    train_new_num_corr = train_new_num.corr()['SalePrice'][
                                         :-1]  # -1 because the latest row is SalePrice
                    close_correlated_features_list = train_new_num_corr[abs(train_new_num_corr) > 0.5].sort_values(
                        ascending=False)
                    print("There is {} strongly correlated values with SalePrice:\n{}".format(
                        len(close_correlated_features_list), close_correlated_features_list))
                    train_correlated = train_new[
                        ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
                         'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'SalePrice']]
                    train_correlated.head()
                    for i in range(0, len(train_correlated.columns), 10):
                        sns.pairplot(data=train_correlated,
                                     x_vars=train_correlated.columns[i:i + 10],
                                     y_vars=['SalePrice'])
                        X = train_correlated.drop(columns='SalePrice', axis=1)
                        Y = train_correlated['SalePrice']
                        X.head()
                        Y.head()
                        model = LogisticRegression(solver="liblinear").fit(X, Y)
                        test = pd.read_csv('test.csv')
                        test.head()
                        predictor_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
                                             '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
                        test_X = test[predictor_columns]
                        test_X.head()
                        test_X.info()
                        med = test_X['GarageCars'].median()
                        test_X['GarageCars'] = test_X['GarageCars'].fillna(med)
                        med = test_X['GarageArea'].median()
                        test_X['GarageArea'] = test_X['GarageArea'].fillna(med)
                        med = test_X['TotalBsmtSF'].median()
                        test_X['TotalBsmtSF'] = test_X['TotalBsmtSF'].fillna(med)
                        test_X.info()
                        predicted_prices = model.predict(test_X)
                        print(predicted_prices)
                        my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
                        my_submission.to_csv('submission.csv', index=False)
                        plt.show()