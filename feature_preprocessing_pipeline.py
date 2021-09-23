# coding=utf-8

import sys
from os.path import join, isfile
from matplotlib.pyplot import title
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

from sklearn.preprocessing import PolynomialFeatures

import bs_lib.bs_transformer as tsf
import bs_lib.bs_preprocess_lib as bsp
import bs_lib.bs_terminal as terminal

from scipy.stats import zscore
from sklearn.impute import KNNImputer
from bs_lib.bs_eda import get_numerical_columns, get_categorical_columns, train_val_test_split

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
    print("\nCreating Columns transformers")
    transformer = ColumnTransformer(
        [
            ("poly", PolynomialFeatures(degree=2,
                                        interaction_only=False,
                                        include_bias=False),
             make_column_selector(dtype_include=np.number)),

            ("mpg_discretizer", KBinsDiscretizer(n_bins=6,
                                                 encode='onehot', strategy='uniform'), ['mpg']),

            ("tax_discretizer", KBinsDiscretizer(n_bins=9,
                                                 encode='onehot', strategy='quantile'), ['tax']),

            ("engine_size_discretizer", KBinsDiscretizer(n_bins=3,
                                                         encode='onehot', strategy='uniform'), ['engine_size']),

            ('year_pipe', Pipeline(steps=[('discretize', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile'))],
                                   verbose=True), ['year']),

            ('model_pipe', Pipeline(steps=[('OHE', OneHotEncoder(
                handle_unknown='ignore'))], verbose=True), ['model']),
            ('brand_pipe', Pipeline(steps=[('OHE', OneHotEncoder(
                handle_unknown='ignore'))], verbose=True), ['brand']),
            ('transmission_pipe', Pipeline(steps=[('OHE', OneHotEncoder(
                handle_unknown='ignore'))], verbose=True), ['transmission']),
            ('fuel_type_pipe', Pipeline(steps=[('OHE', OneHotEncoder(
                handle_unknown='ignore'))], verbose=True), ['fuel_type'])
        ], remainder='passthrough', verbose=2)
    return transformer


def extract_features(data):
    print("\nExtract Features")
    X = data.copy()
    X['age'] = X['year'].max()-X['year']
    X.loc[X['age'] < 1, 'age'] = 1
    # X.drop(['year'],axis=1,inplace=True)
    m_a = X['mileage']/X['age']
    X['mileage_per_year'] = m_a
    mpg_a = X['mpg']/X['age']
    X['mpg_per_year'] = mpg_a
    t_a = X['tax']/X['age']
    X['tax_per_year'] = t_a
    e_a = X['engine_size']/X['age']
    X['engine_per_year'] = e_a
    mmte = X['mileage']+X['mpg']+X['tax']+X['engine_size']
    X['mpy_mpy'] = m_a/mmte+mpg_a/mmte+t_a/mmte+e_a/mmte
    #X.drop('age',axis=1, inplace=True)
    #X['galon_per_year'] = X['mpg']/X['mileage_per_year']
    #X['galon_per_year'] = X['mileage_per_year']/X['mpg']
    #X.drop('mileage_per_year',axis=1, inplace=True)
    #X['tax_per_mileage'] = X['tax']/X['mileage']
    #X['tax_per_mileage'] = X['mileage']/X['tax']
    #X['litre_per_mileage'] = X['engine_size']/X['mileage']
    #X['litre_per_mileage'] = X['mileage']/X['engine_size']
    # X['litre_per_galon'] = X['engine_size']/X['galon_per_year']
    return X


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    return np.exp(y_pred), np.exp(y_val), rmse


def evaluate_prediction(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    y_pred = np.exp(y_pred)
    y_val = np.exp(y_val)

    columns = []
    columns.append(terminal.create_column('Prediction'))
    columns.append(terminal.create_column('Real Price'))
    columns.append(terminal.create_column('Error'))
    data = []
    for i in range(len(y_pred)):
        row = []
        pred = y_pred[i:i+1][0]
        real = int(y_val[i:i+1].values[0])
        error = np.abs(real-pred)
        row.append(f"{pred:.0f}")
        row.append(f"{real}")
        row.append(f"{error}")
        data.append(row)
    table = terminal.create_table(title="Prediction",
                                  columns=columns,
                                  data=data)
    terminal.article(title="Model Evaluation", content=table)


def fit_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    evaluate(model, X_val, y_val)


def get_best_estimator(model, param_grid, X, y, scoring):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=2,
                        n_jobs=-1, pre_dispatch=1
                        )
    grid.fit(X, y)
    print(grid.best_params_)
    return grid.best_estimator_


def get_data(file_name, target_column, dataset_directory_path='./dataset/', sample=None, callback=None, prefix='saved', show=False):
    data = load_data(file_name=file_name, dataset_directory_path=dataset_directory_path,
                     callback=callback, prefix=prefix, show=show)
    if sample:
        data = data.sample(n=sample)
    return get_features_target(data, target_name=target_column, show=show)


def get_features_target(data, target_name, show=False):
    features = data.drop(labels=[target_name], axis=1)
    target = np.log(data[target_name])  # plus normalisation
    #target=target.values.reshape(target.shape[0], 1)
    # target.values.reshape(-1,1)
    if show:
        print(f"X: {features.shape}\ny: {target.shape}")
    return features, target


def load_data(file_name, dataset_directory_path='./dataset/', callback=None, prefix='saved', show=False):
    print(f'Loading {file_name} Dataset...')
    saved_file_name = f"{prefix}_{file_name}"
    saved_file_path = join(dataset_directory_path, saved_file_name)
    if isfile(saved_file_path):
        data = pd.read_csv(saved_file_path, index_col=0)
        if show:
            print(f"{file_name} already processed, {saved_file_name} loaded")
    else:
        file_path = join(dataset_directory_path, file_name)
        data = pd.read_csv(file_path, index_col=0)
        for c in callback:
            data = c(data)
        data.to_csv(saved_file_path)
        if show:
            print(
                f"{file_name} loaded\nA new file has been saved as '{saved_file_name}'")
    return data


def clean_variables(data):
    print("\n\tRemoving dupplicate entries and noisy features:")
    df = data.copy()
    print("\t\tChange type to Categorical")
    columns = get_categorical_columns(df)
    for c in columns:
        df[c] = pd.Categorical(df[c], categories=df[c].unique().tolist())
    # remove duplicate
    print("\t\tDrop duplicate")
    df.drop_duplicates(inplace=True)
    # scale
    print("\t\tMinMax Scaling numerical feature")
    num_columns = get_numerical_columns(df)
    num_columns.remove('price')
    scaler = MinMaxScaler()
    df[num_columns] = scaler.fit_transform(df[num_columns])
    # remove unhandled categories
    print("\t\tRemove unhandled categories")
    df = df[df['transmission'] != 'Other']
    df = df[(df['fuel_type'] != 'Other')]
    df = df[(df['fuel_type'] != 'Electric')]
    return df


def drop_outliers(data):
    return outliers_transformer(data, drop=True)


def nan_outliers(data):
    return outliers_transformer(data)


def outliers_transformer(data, drop=False):
    print('\n\tTransform outliers')
    df = data.copy()
    columns = get_numerical_columns(df)
    columns.remove('price')
    thresh = 3
    if drop:
        outliers = df[columns].apply(lambda x: np.abs(
            zscore(x, nan_policy='omit')) > thresh).any(axis=1)
        print(f"\t\tDroping {outliers.shape[0]} outliers")
        df.drop(df.index[outliers], inplace=True)
    else:
        outliers = df[columns].apply(lambda x: np.abs(
            zscore(x, nan_policy='omit')) > thresh)
        # replace value from outliers by nan
        # print(outliers)
        print(f"\t\ttagging {outliers.shape[0]} outliers")
        for c in outliers.columns.to_list():
            df.loc[outliers[c], c] = np.nan
        print(df.info())
    return df


def numerical_imputer(data, n_neighbors=10, weights='uniform'):
    print('\n\tImput numerical outliers')
    df = data.copy()
    columns = get_numerical_columns(df)
    has_nan = df.isnull().values.any()
    print(f"\t\t{columns} has NAN? {has_nan}")
    if(has_nan):
        print("\t\timputing ...")
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df[columns] = imputer.fit_transform(df[columns])
        print(df)
    # df.replace(np.inf, np.nan)
    # df.replace(-np.inf, np.nan)
    print("\t\tImputation done?", not df.isnull().values.any())
    print("\t\thas inf?", df.isin([np.nan, np.inf, -np.inf]).values.any())
    return df


def categorical_numerizer(data):
    print("\tNumerize categorical data")
    df = data.copy()
    columns = get_categorical_columns(df)
    df[columns] = pd.Categorical(df[columns]).codes
    # df[columns] = df[columns].cat.codes
    # df = df.apply(lambda series: pd.Series(
    #     LabelEncoder().fit_transform(series[series.notnull()]),
    #     index=series[series.notnull()].index
    # ))
    return df


def categorical_imputer(data):
    print('\tImput categorical outliers')
    df = data.copy()
    columns = get_categorical_columns(df)
    df = categorical_numerizer(df)
    thresh = 3
    outliers = df[columns].apply(lambda x: np.abs(
        zscore(x, nan_policy='omit')) > thresh)
    # replace value from outliers by nan
    df[outliers] = np.nan
    has_nan = df.isnull().values.any()
    print(f"\t{columns} has NAN? {has_nan}")
    if(has_nan):
        imputer = KNNImputer(n_neighbors=1, weights='uniform')
        columns = get_numerical_columns(df)
        df[columns] = pd.DataFrame(
            imputer.fit_transform(df[columns]), columns=columns)
    print("\tImputation done?", not df.isnull().values.any())
    return df


def check_integrity(matrix):
    for i in range(len(matrix.data)):
        if np.isnan(matrix.data[i]):
            print("Here it is: ", i, matrix.data[i])
    # np.nan_to_num(matrix.data)
    # print(type(matrix),matrix.data)
    print("Still has nan?", np.any(np.isnan(matrix.data)))
    print("all finite?", np.all(np.isfinite(matrix.data)))
    return matrix  # data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]


if __name__ == "__main__":
    np.random.seed(1)

    # print(sys.version_info)

    dataset_directory_path = './dataset/'
    model_directory_path = 'model/'
    all_data_file = 'all_set.csv'
    train_set_file = 'train_set.csv'
    val_set_file = 'val_set.csv'
    prefix = 'prepared'

    all_data = load_data(all_data_file, dataset_directory_path,
                         callback=[clean_variables, nan_outliers],
                         prefix=prefix, show=True)

    X, y = get_features_target(all_data, target_name='price', show=True)

    X_train, X_val, X_test, y_train, y_val, Y_test = train_val_test_split(X=X,
                                                                          y=y,
                                                                          train_size=.75,
                                                                          val_size=.15,
                                                                          test_size=.1,
                                                                          random_state=1,
                                                                          show=True,
                                                                          export=True,
                                                                          directory_path=dataset_directory_path)
    # *** Modify train_set for training *** #
    X_train_filename = 'X_train.csv'
    X_train_file = f"{prefix}_{X_train_filename}"
    X_train_file_path = join(dataset_directory_path, X_train_file)
    if isfile(X_train_file_path):
        # print(X_train_file_path)
        X_train = load_data(file_name=X_train_filename,
                            dataset_directory_path=dataset_directory_path,
                            prefix=prefix)
        # print(X_train.info())
    else:
        X_train = numerical_imputer(
            X_train, n_neighbors=25, weights='distance')
        X_train.to_csv(X_train_file_path)

    transformer = get_transformer(X_train)

    # *** testing *** #
    # X_ = pd.DataFrame(transformer.fit_transform(X).toarray())
    # #print(X_.info())

    # if model not already exists:
    model_name = 'RFR_1829'
    model_filename = f'{model_name}.joblib'
    model_path = join(model_directory_path, model_filename)
    nb_estimators = 100
    if isfile(model_path):
        model = load(model_path)
        # print(model)
    else:
        # pipeline: predict preprocessing
        steps = [
            ("features_extraction", FunctionTransformer(
                extract_features, validate=False)),
            ("transformer", transformer),
            ("scaler", StandardScaler(with_mean=False)),
            # ("MinMax", MaxAbsScaler()),
            #("check integrity",FunctionTransformer(check_integrity)),
            ("random_forest", RandomForestRegressor(
                n_estimators=nb_estimators,
                max_features='auto',
                min_samples_split=6,
                max_depth=50,
                n_jobs=-1, verbose=True, warm_start=True))
        ]
        model = Pipeline(steps=steps, verbose=True)

        model.fit(X_train, y_train)
        dump(model, model_path)

    param_grid = {
        'random_forest__max_depth': [40, 50, 100],
        'random_forest__min_samples_split': np.arange(2, 8, 2),
        'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
        # {'random_forest__max_depth': 50,
        # 'random_forest__max_features': 'auto',
        # 'random_forest__min_samples_split': 6}

        'transformer__poly__degree': [2, 3],
        'transformer__poly__interaction_only': [True, False],
        'transformer__poly__include_bias': [True, False],
        #   {'transformer__poly__degree': 2,
        #   'transformer__poly__include_bias': False,
        #   'transformer__poly__interaction_only': False}

        'transformer__mpg_discretizer__n_bins': [5, 6, 10, 13],
        'transformer__mpg_discretizer__encode': ['onehot', 'ordinal'],
        'transformer__mpg_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
        #   {'transformer__mpg_discretizer__encode': 'onehot',
        #   'transformer__mpg_discretizer__n_bins': 6,
        #   'transformer__mpg_discretizer__strategy': 'uniform'}

        'transformer__tax_discretizer__n_bins': [7, 8, 9, 10],
        'transformer__tax_discretizer__encode': ['onehot', 'ordinal'],
        'transformer__tax_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
        #   {'transformer__tax_discretizer__encode': 'onehot',
        #   'transformer__tax_discretizer__n_bins': 9, '
        #   transformer__tax_discretizer__strategy': 'quantile'}

        'transformer__engine_size_discretizer__n_bins': [2, 3, 4, 6, 9],
        'transformer__engine_size_discretizer__encode': ['onehot', 'ordinal'],
        'transformer__engine_size_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
        #   {'transformer__engine_size_discretizer__encode': 'onehot',
        #   'transformer__engine_size_discretizer__n_bins': 3,
        #   'transformer__engine_size_discretizer__strategy': 'uniform'}

        'transformer__year_pipe__discretize__n_bins': [3, 6, 9, 11],
        'transformer__year_pipe__discretize__encode': ['onehot', 'ordinal'],
        'transformer__year_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans']
        #   {'transformer__year_pipe__discretize__encode': 'onehot',
        #   'transformer__year_pipe__discretize__n_bins': 11,
        #   'transformer__year_pipe__discretize__strategy': 'uniform'}
    }
    # *** GridSearchCV ***#
    # mse = make_scorer(mean_squared_error, greater_is_better=False)
    # model = get_best_estimator(
    #     model, param_grid, X_train, y_train, scoring=mse)
    # dump(model, model_path)

    evaluate(model, X_val, y_val)

    #RMSE: 1829.6801093112176

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
