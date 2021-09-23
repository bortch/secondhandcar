# coding=utf-8

import sys
from os import rename
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

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from bs_lib.bs_eda import load_csv,load_all_csv
from bs_lib.bs_eda import get_numerical_columns, get_categorical_columns, train_val_test_split, split_by_row, load_csv_files_as_dict

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
            # ("poly", PolynomialFeatures(degree=2,
            #                             interaction_only=False,
            #                             include_bias=False),
            #  make_column_selector(dtype_include=np.number)),

            # ("mpg_discretizer", KBinsDiscretizer(n_bins=6,
            #                                      encode='onehot', strategy='uniform'), ['mpg']),

            # ("tax_discretizer", KBinsDiscretizer(n_bins=9,
            #                                      encode='onehot', strategy='quantile'), ['tax']),

            # ("engine_size_discretizer", KBinsDiscretizer(n_bins=3,
            #                                              encode='onehot', strategy='uniform'), ['engine_size']),

            # ('year_pipe', Pipeline(steps=[('discretize', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile'))],
            #                        verbose=True), ['year']),

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
    X = data.copy()
    X['age'] = X['year'].max()-X['year']
    X.loc[X['age'] < 1, 'age'] = 1
    # X.drop(['year'],axis=1,inplace=True)
    m_a = X['mileage']/X['age']
    #X['mileage_per_year'] = m_a
    mpg_a = X['mpg']/X['age']
    #X['mpg_per_year'] = mpg_a
    t_a = X['tax']/X['age']
    #X['tax_per_year'] = t_a
    e_a = X['engine_size']/X['age']
    #X['engine_per_year'] = e_a
    mmte = X['mileage']+X['mpg']+X['tax']+X['engine_size']
    #X['mpy_mpy'] = m_a/mmte+mpg_a/mmte+t_a/mmte+e_a/mmte
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
    print(f"\nModel Evaluation")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    return np.exp(y_pred), np.exp(y_val), rmse


def evaluate_prediction(model, X_val, y_val, sample = None):
    if sample: 
        X_val = X_val.sample(n=sample, random_state=1)
        y_val = y_val.sample(n=sample, random_state=1)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    y_pred = np.exp(y_pred)
    y_val = np.exp(y_val)

    data = []

    for i in range(len(y_pred)):
        row = []
        pred = y_pred[i:i+1][0]
        real = int(y_val[i:i+1].values[0])
        error = np.abs((real-pred))
        percentage = error/real*100
        row.append(f"{pred:.0f}")
        row.append(f"{real:.0f}")
        row.append(f"{error:.0f}")
        row.append(f"{percentage:.0f} %")
        data.append(row)
    table = terminal.create_table(title="Prediction results",
                                  columns=['Prediction','Real Price','Error', 'Percentage'],
                                  data=data)
    terminal.article(title="Model Prediction testing", content=table)


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
    print(f'\nLoading {file_name} Dataset\n...')
    saved_file_name = f"{prefix}_{file_name}"
    saved_file_path = join(dataset_directory_path, saved_file_name)
    if isfile(saved_file_path):
        data = pd.read_csv(saved_file_path, index_col=0)
        if show:
            print(f"\t{file_name} was previously processed, {saved_file_name} reloaded")
    else:
        file_path = join(dataset_directory_path, file_name)
        data = pd.read_csv(file_path, index_col=0)
        for c in callback:
            data = c(data)
        data.to_csv(saved_file_path)
        if show:
            print(
                f"\t{file_name} loaded\n\tA new backup file has been saved as '{saved_file_name}'")
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
    print("\t\tScaling numerical feature")
    num_columns = get_numerical_columns(df)
    #df['price']=df['price']/1.
    for c in num_columns:
        df[c] = pd.to_numeric(df[c])
    num_columns.remove('price')
    scaler = MinMaxScaler()
    df[num_columns] = scaler.fit_transform(df[num_columns])
    std_scaler = StandardScaler(with_mean=False)
    df[num_columns] = std_scaler.fit_transform(df[num_columns])
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
    print('\nTransform outliers')
    df = data.copy()
    columns = get_numerical_columns(df)
    columns.remove('price')
    thresh = 3
    if drop:
        outliers = df[columns].apply(lambda x: np.abs(
            zscore(x, nan_policy='omit')) > thresh).any(axis=1)
        print(f"\tDroping outliers")
        df.drop(df.index[outliers], inplace=True)
    else:
        outliers = df[columns].apply(lambda x: np.abs(
            zscore(x, nan_policy='omit')) > thresh)
        # replace value from outliers by nan
        # print(outliers)
        print(f"\ttagging outliers")
        for c in outliers.columns.to_list():
            df.loc[outliers[c], c] = np.nan
        #print(df.info())
    return df


def numerical_imputer(data, n_neighbors=10, weights='uniform'):
    print('\nImput numerical missing value')
    df = data.copy()
    #print("df",df.info())
    columns = get_numerical_columns(df)
    has_nan = df.isnull().values.any()
    print(f"\t{columns} has NAN? {has_nan}")
    if(has_nan):
        print("\timputing ...")
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        #imputer = IterativeImputer(random_state=0)
        imputed = imputer.fit_transform(df[columns])
        for i,c in enumerate(columns):
            df[c]=imputed[:,i]
        print("\thas inf?", df.isin([np.nan, np.inf, -np.inf]).values.any())
    print("\tImputation done?", not df.isnull().values.any())
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
    matrix.data = np.nan_to_num(matrix.data)
    # for i in range(len(matrix.data)):
    #     if np.isnan(matrix.data[i]):
    #         print("Here it is: ", i, matrix.data[i])
    # # np.nan_to_num(matrix.data)
    # # print(type(matrix),matrix.data)
    # print("Still has nan?", np.any(np.isnan(matrix.data)))
    # print("all finite?", np.all(np.isfinite(matrix.data)))
    # #df.columns.to_series()[np.isinf(df).any()]
    return matrix  # data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]


if __name__ == "__main__":
    np.random.seed(1)

    pd.options.mode.use_inf_as_na = True
    # print(sys.version_info)

    dataset_directory_path = './dataset/'
    model_directory_path = 'model/'
    all_data_file = 'all_set.csv'
    train_set_file = 'train_set.csv'
    val_set_file = 'val_set.csv'
    prefix = 'prepared'
    path_to_files = join(dataset_directory_path, prefix)
    
    all_data = load_data(all_data_file, dataset_directory_path,
                         callback=[clean_variables],
                         prefix=prefix, show=True)
    
    #all_data = nan_outliers(all_data)
    #all_data = numerical_imputer(all_data, n_neighbors=50, weights='distance')
    
    all_filename = ['X_train.csv','X_val.csv','X_test.csv','y_train.csv','y_val.csv','y_test.csv']
    if isfile(join(path_to_files,f"{all_filename[0]}")):
        all_files = load_all_csv(path_to_files, index=0)
        X_train = all_files["x_train"]
        X_val = all_files["x_val"]
        X_test = all_files["x_test"]
        y_train = all_files["y_train"]
        y_val = all_files["y_val"]
        y_test = all_files["y_test"]
    else:
        print("\nSplitting into train, val and test sets")
        train_set, test_set = split_by_row(all_data, .8)
        
        # *** Modify train_set for training *** #
        train_set = nan_outliers(train_set)
        train_set = numerical_imputer(train_set, n_neighbors=100, weights='distance')
        
        X_train_val, y_train_val = get_features_target(train_set, target_name='price', show=True)
        X_test, y_test = get_features_target(test_set, target_name='price', show=True)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,train_size=.85,random_state=1)
        print(f"\tX_train: {X_train.shape}\n\tX_val: {X_val.shape}\n\tX_test: {X_test.shape}\n\ty_train: {y_train.shape}\n\ty_val: {y_val.shape}\n\ty_test: {y_test.shape}")

        X_train.to_csv(join(path_to_files,"X_train.csv"))
        y_train.to_csv(join(path_to_files,"y_train.csv"))
        X_val.to_csv(join(path_to_files,"X_val.csv"))
        y_val.to_csv(join(path_to_files,"y_val.csv"))
        X_test.to_csv(join(path_to_files,"X_test.csv"))
        y_test.to_csv(join(path_to_files,"y_test.csv"))

    transformer = get_transformer(X_train)

    # *** testing *** #
    # X_ = pd.DataFrame(transformer.fit_transform(X).toarray())
    # #print(X_.info())

    # if model not already exists:
    temp_model_name = 'temp_model'
    temp_model_filename = f'{temp_model_name}.joblib'
    temp_model_path = join(model_directory_path, temp_model_filename)
    nb_estimators = 10
    model_to_load =''#'model_10_1937.8390856038359.joblib'
    model_to_load_path = join(model_directory_path,model_to_load)
    if isfile(model_to_load_path):
       model = load(model_to_load_path)
       # print(model)
    else:
    #pipeline: predict preprocessing
        steps = [
            ("features_extraction", FunctionTransformer(
                extract_features, validate=False)),
            ("transformer", transformer),
            #("check integrity",FunctionTransformer(check_integrity)),
            ("random_forest", RandomForestRegressor(
                n_estimators=nb_estimators,
                max_features='auto',
                min_samples_split=6,
                max_depth=50,
                n_jobs=-1, 
                #warm_start=True, #Optimise computation during GridSearchCV
                verbose=True
                ))
        ]
        model = Pipeline(steps=steps, verbose=True)

        print("\nTraining the model")
        model.fit(X_train, y_train)
        dump(model, temp_model_path)

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
    print(X_val.isna().any())
    y_pred,y_val,rmse = evaluate(model, X_val, y_val)
    model_name = f'model_{nb_estimators}_{rmse}'
    model_filename = f'{model_name}.joblib'
    model_path = join(model_directory_path, model_filename)
    rename(temp_model_path, model_path)
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
