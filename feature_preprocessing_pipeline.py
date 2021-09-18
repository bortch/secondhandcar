from os.path import join, isfile
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

from sklearn.preprocessing import PolynomialFeatures

import bs_lib.bs_transformer as tsf
import bs_lib.bs_preprocess_lib as bsp

from scipy.stats import zscore
from sklearn.impute import KNNImputer
from bs_lib.bs_eda import get_numerical_columns

from joblib import dump, load

# TODO:
# Preprocessing for All
# [X] 'price', 'mileage' as float
# [X] 'year' as discrete 'year_category'
# [X] 'mpg' as discrete 'mpg_category'
# [X] 'mpg_category' through Ordinal Encoder
# [X] 'engine_size' as discrete 'engine_category'
# [X] 'engine_category' through  Ordinal Encoder
# [X] 'tax' as discrete 'tax_category'
# [ ] merge 'fuel_type'
# [X] One Hot Encoding 'model' & 'brand'
# PCA
# [X] Standard Scaling


def categorize(X):
    X = pd.DataFrame(X).copy()
    columns = X.select_dtypes(include=['object']).columns.tolist()
    for c in columns:
        #print(f"\nCategorise: {c}\n")
        X[c] = X[c].astype('category')
        X[c] = X[c].cat.codes
    return X


def discretize(X, kw_args):
    #print(f"\nDiscretize: {X.columns.to_list()}\n")
    return X.apply(pd.cut, **kw_args)  # .cat.codes)


def get_transformer(X):

    # categorizer = FunctionTransformer(categorise)
    # year_bins = np.arange(2009, 2022)
    # mpg_bins = [0, 36, 47, 100]
    # engine_bins = [-1, 2, 7]
    # tax_bins = [-1, 100, 125, 175, 225, 250, 275, 1000]

    categorical_pipeline = Pipeline(steps=[('Categorizer', FunctionTransformer(categorize)),
                                           ('OHE', OneHotEncoder(handle_unknown='ignore'))],
                                    verbose=True)

    categorical_ordinal_pipeline = Pipeline(steps=[('Categorizer', FunctionTransformer(categorize)),
                                                   ('Ordinal Encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan))],
                                            verbose=True)

    poly_transformer = Pipeline(steps=[('Polynomial Features', PolynomialFeatures(degree=3,
                                                                                  interaction_only=True,
                                                                                  include_bias=False))
                                       #                                               ,
                                       #    ('std Scaler', StandardScaler())
                                       ],
                                verbose=True
                                )

    transformer = ColumnTransformer(
        [
            ("Poly features Creator", poly_transformer,
             make_column_selector(dtype_include=np.number)),
            ("'mileage' to Float Converter",
             tsf.TypeConverter('float'), ['mileage']),
            ("'mpg' Discretizer", KBinsDiscretizer(n_bins=6,
                                                   encode='onehot', strategy='kmeans'), ['mpg']),
            ("'engine_size' Discretizer", KBinsDiscretizer(n_bins=4,
                                                           encode='ordinal', strategy='kmeans'), ['engine_size']),
            ("'tax' Discretizer", KBinsDiscretizer(n_bins=9,
                                                   encode='ordinal', strategy='kmeans'), ['tax']),
            ("Drop Colinear Variables", 'drop', ['model', 'year']),
            ("'Transmission', 'Fuel' Ordinal Encoder",
             categorical_ordinal_pipeline, ['transmission', 'fuel_type']),
            ("'Brand' OHE", categorical_pipeline, ['brand'])
        ], remainder='passthrough', verbose=True)

    # transformer = make_column_transformer(
    #     (poly_transformer, make_column_selector(dtype_include=np.number)),
    #     (tsf.TypeConverter('float'), ['mileage']),
    #     (KBinsDiscretizer(n_bins=6,
    #      encode='onehot', strategy='kmeans'), ['mpg']),
    #     (KBinsDiscretizer(n_bins=4,
    #                       encode='ordinal', strategy='kmeans'), ['engine_size']),
    #     (KBinsDiscretizer(n_bins=9,
    #      encode='ordinal', strategy='kmeans'), ['tax']),
    #     #(categorizer, ['transmission', 'fuel_type']),
    #     ('drop', ['model', 'year']),
    #     (categorical_ordinal_pipeline, ['transmission', 'fuel_type']),
    #     (categorical_pipeline, ['brand']),
    #     remainder='passthrough', verbose=True)
    return transformer


def extract_features(data):
    X = data.copy()
    X['age'] = X['year'].max()-X['year']
    X.loc[X['age'] < 1, 'age'] = 1
    X['mileage_per_year'] = X['mileage']/X['age']
    #X.drop('age',axis=1, inplace=True)
    X['galon_per_year'] = X['mpg']/X['mileage_per_year']
    #X['galon_per_year'] = X['mileage_per_year']/X['mpg']
    X['tax_per_mileage'] = X['tax']/X['mileage']
    X['litre_per_mileage'] = X['engine_size']/X['mileage']
    X['litre_per_galon'] = X['engine_size']/X['galon_per_year']
    return X


def clean_variables(data):
    df = data.copy()
    # remove duplicate
    df.drop_duplicates(inplace=True)
    # remove unhandled categories
    df = df[df['transmission'] != 'Other']
    df = df[(df['fuel_type'] != 'Other')]
    df = df[(df['fuel_type'] != 'Electric')]
    return df


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    return np.exp(y_pred), np.exp(y_val)


def evaluate_prediction(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    y_pred = np.exp(y_pred)
    y_val = np.exp(y_val)
    print("prediction \t| real price")
    for i in range(len(y_pred)):
        print(f"{y_pred[i:i+1][0]:.0f} \t\t| {int(y_val[i:i+1].values[0])}")


def fit_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    evaluate(model, X_val, y_val)


def get_best_estimator(model, param_grid, X_train, y_train, scoring):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=1, n_jobs=-1
                        )
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    return grid.best_estimator_


def get_data(file_path, target_column, dataset_directory_path='./dataset/', sample=None, callback=None):
    data = pd.read_csv(join(dataset_directory_path, file_path), index_col=0)
    for c in callback:
        data = c(data)
    if sample:
        data = data.sample(n=sample)
    X = data.drop(labels=[target_column], axis=1)
    y = np.log(data[target_column])  # plus normalisation
    return X, y

def x_or_nan(x,thresh):
    # mask of all rows whose value > thresh
    if np.abs(zscore(x, nan_policy = 'omit')) > thresh:
        return np.nan
    else:
        return x

def prepare_data(data):
    df = data.copy()
    columns = get_numerical_columns(df)
    thresh = 3
    outliers = df[columns].apply(lambda x: np.abs(zscore(x, nan_policy = 'omit')) > thresh)
    # replace value from outliers by nan
    df[outliers] = np.nan
    
    imputer = KNNImputer(weights='distance')
    #columns = get_numerical_columns(df)
    df[columns] = pd.DataFrame(imputer.fit_transform(df[columns]),columns=columns)
    return df

if __name__ == "__main__":
    np.random.seed(1)

    dataset_directory_path = 'dataset/'
    model_directory_path = 'model/'
    train_set_file = 'train_set.csv'
    val_set_file = 'test_set.csv'
    train_data = pd.read_csv(
        join(dataset_directory_path, train_set_file), index_col=0)

    #X_train = train_data.drop(labels=['price'], axis=1)
    # y_train = np.log(train_data['price']) # Target + Normalisation
    X_train, y_train = get_data(file_path=train_set_file, target_column='price',
                                dataset_directory_path=dataset_directory_path, 
                                sample=None, callback=[clean_variables,prepare_data])
    X_val, y_val = get_data(file_path=val_set_file, target_column='price',
                            dataset_directory_path=dataset_directory_path, 
                            sample=None, callback=[clean_variables])
    #val_data = pd.read_csv(join(dataset_directory_path, val_set_file), index_col=0)
    #X_val = val_data.drop(labels=['price'], axis=1)
    #y_val = np.log(val_data['price'])

    transformer = get_transformer(X_train)

    # testing
    # X_ = pd.DataFrame(transformer.fit_transform(X).toarray())
    # #print(X_.info())

#    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.15)

    # get_features = FunctionTransformer(extract_features, validate=False)
    # get_cleaned_variables = FunctionTransformer(clean_variables)
    # scaler = ColumnTransformer([("Scaler",
    #                              StandardScaler(),
    #                              make_column_selector(dtype_include=np.number))],
    #                              verbose=True, remainder='passthrough')
    # polygenerator = ColumnTransformer([("Poly features generator",
    #                                     PolynomialFeatures(degree=3,
    #                                                        interaction_only=True,
    #                                                        include_bias=False),
    #                                     make_column_selector(dtype_include=np.number))],
    #                                     verbose=True, remainder='passthrough')

    # model = make_pipeline(get_features,
    #                       transformer,
    #                       RandomForestRegressor(
    #                           n_estimators=200, n_jobs=-1),
    #                       verbose=True)
    # model.fit(X_train, y_train)
    # evaluate(model, X_val, y_val)

    # if model not already exists:
    model_name = 'model_RMSE-678'
    model_filename = f'{model_name}.joblib'
    model_path = join(model_directory_path, model_filename)
    nb_estimators = 500
    if isfile(model_path):
        model = load(model_path)
        # print(model)
    else:
        # pipeline: predict preprocessing
        steps = [
            #('Clean Variables',FunctionTransformer(clean_variables)),
            ("Features Extraction", FunctionTransformer(
                extract_features, validate=False)),
            ("Columns Transformer", transformer),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=nb_estimators, n_jobs=-1))
            ]
        model = Pipeline(steps=steps, verbose=True)

        # model = make_pipeline(get_features,
        #                       transformer,
        #                       RandomForestRegressor(
        #                           n_estimators=500, n_jobs=-1),
        #                       verbose=True)

        model.fit(X_train, y_train)
        dump(model, model_path)

    param_grid = {'randomforestregressor__max_depth': [40, 50, 100],
                  'randomforestregressor__min_samples_split': np.arange(2, 8, 2)
                  }

    # Redefine Scoring
    # model = make_pipeline(transformer,
    #                       RandomForestRegressor(n_estimators=200, n_jobs=-1),
    #                       verbose=False)
    # mse = make_scorer(mean_squared_error, greater_is_better=False)
    # model = get_best_estimator(
    #     model, param_grid, X_train, y_train, scoring=mse)

    evaluate(model, X_val, y_val)

    # RMSE: 908.2924800638677
    # drop [Model] : RMSE: 775.1466069992655
    # drop [Brand] : RMSE: 872.0164550638308
    # drop [Model], Engine_size {bins:6,ordinal}: RMSE: 774.8210885246958
    # drop [Model], Engine_size {bins:3,ordinal}: RMSE: 775.483119993178
    # drop [Model], Engine_size {bins:4,ordinal}: RMSE: 774.9931127333747
    # drop [Model], Engine_size {bins:4,onehot}: RMSE: 774.7801325578779
    # drop [Model], Engine_size {bins:4,onehot},mpg{bins:6,ordinal}: RMSE: 775.1856423450347
    # drop [Model], Engine_size {bins:4,onehot},mpg{bins:6,ordinal},tax{bins:9,ordinal}: RMSE: 774.9598569675812
    # drop [Model], Engine_size {bins:4,onehot},mpg{bins:6,onehot},tax{bins:9,ordinal}: RMSE: 774.8791007133806
    # add miles_per_year: RMSE: 770.1800524934332
    # previous *_per_* feature + galon_per_year: RMSE: 752.0092177388406
    # previous *_per_* features + tax_per_mileage: RMSE: 735.1036011216089
    # previous *_per_* features + litre_per_mileage: RMSE: 710.4270104501544
    # previous *_per_* features + litre_per_galon: RMSE: 708.0328930831951
    # polynomiale feature 2 without *_per_* features RMSE: 730.0088958446174
    # polynomiale feature 3 without *_per_* features RMSE: 723.2213099241444
    # polynomiale feature 3 + bias, with *_per_* features  RMSE: 679.4338384834626
    # polynomiale feature 3 without biais, with *_per_* features RMSE: 678.1993773405426
    # polynomiale feature 3 without biais, with *_per_* features + age RMSE: 678.5200863970292
    # polynomiale feature 3 without biais, with *_per_* features - age - year RMSE: 681.0919899454383
    # * polynomiale feature 3 without biais, with *_per_* features + age - year: RMSE: 677.2515811285315
    # poly 3 no biais + *_per_* features + age - year + OHE Fueltype & transmission RMSE: 677.2515811285316 RMSE: 677.2666317999145

    #bsp.get_learning_curve(model, X_train, y_train, scoring='neg_mean_squared_error',show=False,savefig=True)
    #bsp.plot_learning_curve(model, model_name, X_train, y_train, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), show=False, savefig=True)
