# coding=utf-8

from pandas.core.indexes import category
from bs_lib.bs_eda import get_categorical_columns
import sys
from os import rename
from os.path import join, isfile
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
from pandas.core.arrays import sparse
import bs_lib.bs_terminal as terminal

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

from bs_lib.bs_eda import load_csv, load_all_csv
from bs_lib.bs_eda import train_val_test_split

from joblib import dump, load


def get_transformer(transformers=[], verbose=False):
    print("\nCreating Columns transformers")
    transformers_ = [
        # ("poly", PolynomialFeatures(degree=2,
        #                             interaction_only=False,
        #                             include_bias=False),
        #  make_column_selector(dtype_include=np.number)),

        ("mpg_pipe", Pipeline(steps=[
            ('discretize', KBinsDiscretizer(n_bins=6,
                                            encode='onehot', strategy='uniform'))
        ], verbose=verbose), ['mpg']),

        # ("tax_discretizer", KBinsDiscretizer(n_bins=9,
        #                                      encode='onehot', strategy='quantile'), ['tax']),

        ("engine_size_pipe", Pipeline(steps=[
            ('discretize',  KBinsDiscretizer(n_bins=3,
                                             encode='onehot', strategy='uniform'))
        ], verbose=verbose), ['engine_size']),

        ('year_pipe', Pipeline(steps=[
            ('discretize', KBinsDiscretizer(
                n_bins=10, encode='ordinal', strategy='quantile'))
        ], verbose=verbose), ['year']),

        ('model_pipe', Pipeline(steps=[
            #('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['model']),
        
        ('brand_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore'))
        ], verbose=verbose), ['brand']),
        
        ('transmission_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore'))
        ], verbose=verbose), ['transmission']),
        ('fuel_type_pipe', Pipeline(steps=[
            #('OHE', OneHotEncoder(handle_unknown='ignore'))
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['fuel_type'])
    ]
    if transformers:
        transformers_ = transformers_+transformers
        print(transformers_)

    transformer = ColumnTransformer(
        transformers_, remainder='passthrough', verbose=verbose)

    return transformer


def extract_features(data):
    X = data.copy()

    # drop testing
    # ['year', 'price', 'mileage', 'tax', 'mpg', 'engine_size']
    # X.drop(['year'],axis=1,inplace=True)
    # X.drop(['tax'],axis=1,inplace=True)
    # X.drop(['mileage'],axis=1,inplace=True)
    # X.drop(['mpg'],axis=1,inplace=True)
    # X.drop(['engine_size'],axis=1,inplace=True)
    #X.drop(['brand'],axis=1,inplace=True)

    X['model_count'] = X.groupby('model')['model'].transform('count')

    occ = X['model_count']/X.shape[0]
    # adding feature
    X['age'] = X['year'].max()-X['year']
    X.loc[X['age'] < 1, 'age'] = 1
    m_a = X['mileage']/X['age']
    # X['mileage_per_year'] = m_a
    mpg_a = X['mpg']/X['age']
    # X['mpg_per_year'] = mpg_a
    t_a = X['tax']/X['age']
    X['tax_per_year'] = t_a
    e_a = X['engine_size']/X['age']
    #X['engine_per_year'] = e_a
    mmte = (X['mileage']+X['mpg']+X['tax']+X['engine_size'])/occ
    X['mpy_mpy'] = (m_a/mmte+mpg_a/mmte+t_a/mmte+e_a/mmte)
    #X.drop('age',axis=1, inplace=True)
    #X['galon_per_year'] = X['mpg']/X['mileage_per_year']
    #X['galon_per_year'] = X['mileage_per_year']/X['mpg']
    #X.drop('mileage_per_year',axis=1, inplace=True)
    #X['tax_per_mileage'] = X['tax']/X['mileage']
    #X['tax_per_mileage'] = X['mileage']/X['tax']
    #X['litre_per_mileage'] = X['engine_size']/X['mileage']
    #X['litre_per_mileage'] = X['mileage']/X['engine_size']
    #X['litre_per_galon'] = X['engine_size']/X['galon_per_year']
    return X


def evaluate(model, X_val, y_val):
    print(f"\nModel Evaluation")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(f"RMSE: {rmse}")
    return np.exp(y_pred), np.exp(y_val), rmse


def evaluate_prediction(model, X_val, y_val, sample=None):
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
                                  columns=['Prediction', 'Real Price',
                                           'Error', 'Percentage'],
                                  data=data)
    terminal.article(title="Model Prediction testing", content=table)


def print_performance(data):
    table = terminal.create_table(title="One model by Brand",
                                  columns=['Brand', 'RMSE', 'Target mean', 'Prediction mean', 'Model path'], data=data)
    terminal.article(title="Performance Comparison", content=table)


def fit_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    evaluate(model, X_val, y_val)


def get_best_estimator(model, param_grid, X, y, scoring, verbose=False):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=verbose,
                        n_jobs=-1, pre_dispatch=1
                        )
    grid.fit(X, y)
    print(grid.best_params_)
    return grid.best_estimator_


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


def get_model(model_path_to_load=None, verbose=False, warm_start=False):
    #add_transformers = get_categorical_pipeline(categories=categories)
    transformers = get_transformer()  # (transformers=add_transformers)
    nb_estimators = 10
    if (model_path_to_load is not None) and isfile(model_path_to_load):
        model = load(model_path_to_load)
    else:
        steps = [
            ("features_extraction", FunctionTransformer(
                extract_features, validate=False)),
            ("transformer", transformers),
            #("check integrity",FunctionTransformer(check_integrity)),
            ("random_forest", RandomForestRegressor(
                n_estimators=nb_estimators,
                max_features=None,
                min_samples_split=6,
                max_depth=50,
                n_jobs=-1,
                warm_start=warm_start,  # Optimise computation during GridSearchCV
                verbose=verbose
            ))
        ]
    return Pipeline(steps=steps, verbose=verbose)


def get_all_models(files_directory, target, dump_model=False, model_directory='', verbose=False):

    all_df = load_all_csv(dataset_path=files_directory, index=0)

    report = []

    for brand, df in all_df.items():
        df_target = df[target]
        categories = get_ordered_categories(df, by='price')
        df = df.drop(target, axis=1)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=df,
                                                                              y=df_target,
                                                                              train_size=.75,
                                                                              val_size=.15,
                                                                              test_size=.1,
                                                                              random_state=1,
                                                                              show=verbose)

        model = get_model()
        model.set_params(**{
            "transformer__model_pipe__OE__categories": [categories['model']],
            "transformer__brand_pipe__OHE__categories": [categories['brand']],
            "transformer__transmission_pipe__OHE__categories": [categories['transmission']],
            "transformer__fuel_type_pipe__OE__categories": [categories['fuel_type']],
        })
        print(f"\nTraining the model for {brand}")
        model.fit(X_train, y_train)
        model_path = '-'
        if dump_model:
            temp_model_name = f'temp_{brand}_model'
            temp_model_filename = f'{temp_model_name}.joblib'
            temp_model_path = join(model_directory_path, temp_model_filename)
            dump(model, temp_model_path)
            print(f"Model {brand}:\n")

        y_pred, y_val, rmse = evaluate(model, X_val, y_val)

        if dump_model:
            model_name = f'model_{brand}_{round(rmse,0)}'
            model_filename = f'{model_name}.joblib'
            model_path = join(model_directory_path, model_filename)
            rename(temp_model_path, model_path)

        report.append([brand.title(),
                       int(round(rmse, 0)),
                       int(round(y_val.mean(), 0)),
                       int(round(y_pred.mean(), 0)),
                       model_path])

    return report


def get_ordered_categories(df, by):
    categories = {}
    columns = get_categorical_columns(df)
    for cat in columns:
        ordered_df = df[[cat, by]]
        ordered_df = ordered_df.groupby(cat).agg('mean').reset_index()
        ordered_df.sort_values(
            by, ascending=True, inplace=True, ignore_index=True)
        categories[cat] = ordered_df[cat].values
        #print(categories[cat])
    return categories


def get_all_categories(all_df):
    categories = {}
    for brand, df in all_df.items():
        columns = get_categorical_columns(df)
        for cat in columns:
            # keep data to sort
            ordered_df = df[[cat, 'price']]
            # for current brand get aggregated mean by category
            ordered_df = ordered_df.groupby(cat).agg('mean').reset_index()
            # memoize result in categories object
            if cat in categories:
                # append new result to all results by category (all brand merged)
                categories[cat] = categories[cat].append(
                    [categories[cat], ordered_df], ignore_index=True)
                # recompute mean and aggregate for each category
                categories[cat] = categories[cat].groupby(
                    cat).agg('mean').reset_index()
            else:
                categories[cat] = ordered_df

    results = {}  # ordered results
    for cat, df in categories.items():
        # sort each categories
        df.sort_values("price", ascending=True,
                       inplace=True, ignore_index=True)
        results[cat] = df[cat].values
        # print(results[cat])
    return results


def get_one_model_for_all(files_directory, target, dump_model=False, model_directory='', verbose=False):
    all_df = load_all_csv(dataset_path=files_directory, index=0)
    categories_dict = get_all_categories(all_df)
    report = []

    model = get_model(warm_start=True)
    model.set_params(**{
        "transformer__model_pipe__OE__categories": [categories_dict['model']],
        "transformer__brand_pipe__OHE__categories": [categories_dict['brand']],
        "transformer__transmission_pipe__OHE__categories": [categories_dict['transmission']],
        "transformer__fuel_type_pipe__OE__categories": [categories_dict['fuel_type']],
    })
    if dump_model:
        temp_model_name = f'temp_all_brand_model'
        temp_model_filename = f'{temp_model_name}.joblib'
        temp_model_path = join(model_directory_path, temp_model_filename)
        dump(model, temp_model_path)
        print(f"Model for all brand:\n")
    n_estimators = 10
    columns = ['model', 'year', 'transmission',
               'mileage', 'fuel_type',
               'tax', 'mpg',
               'engine_size', 'brand']
    all_X_val = pd.DataFrame(columns=columns)
    all_y_val = pd.Series()
    i = 0
    print(f"\nTraining the model with:")
    for brand, df in all_df.items():
        df_target = df[target]
        df = df.drop(target, axis=1)
        # df.drop(['model','brand','transmission','fuel_type'],axis=1,inplace=True)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=df,
                                                                              y=df_target,
                                                                              train_size=.75,
                                                                              val_size=.15,
                                                                              test_size=.1,
                                                                              random_state=1,
                                                                              show=verbose)
        print(f"\t{brand}")
        estimators = n_estimators+i*n_estimators
        model.set_params(**{"random_forest__n_estimators": estimators})
        model.fit(X_train, y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        i += 1

    y_pred, y_val, rmse = evaluate(model, X_val, y_val)

    model_path = '-'
    if dump_model:
        model_name = f'model_all_brand_{round(rmse,0)}'
        model_filename = f'{model_name}.joblib'
        model_path = join(model_directory_path, model_filename)
        rename(temp_model_path, model_path)

    report.append([f"all brand",
                   int(round(rmse, 0)),
                   int(round(y_val.mean(), 0)),
                   int(round(y_pred.mean(), 0)),
                   model_path])

    return report


if __name__ == "__main__":
    np.random.seed(1)

    pd.options.mode.use_inf_as_na = True

    current_directory = "."
    dataset_directory = "dataset"
    files_directory = join(
        current_directory, dataset_directory, 'prepared_data')

    model_directory_path = 'model/'
    all_data_file = 'all_set.csv'
    train_set_file = 'train_set.csv'
    val_set_file = 'val_set.csv'

    evaluations = get_all_models(files_directory, target='price', verbose=False)
    print_performance(evaluations)

    # ofa_evaluations = get_one_model_for_all(
    #     files_directory, target='price', verbose=False)
    # print_performance(ofa_evaluations)

    param_grid = {
        # 'random_forest__max_depth': [40, 50, 100],
        # 'random_forest__min_samples_split': np.arange(2, 8, 2),
        # 'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
        # {'random_forest__max_depth': 50,
        # 'random_forest__max_features': 'auto',
        # 'random_forest__min_samples_split': 6}

        # 'transformer__poly__degree': [1,2, 3, 4],
        # 'transformer__poly__interaction_only': [True, False],
        # 'transformer__poly__include_bias': [True, False],
        #   {'transformer__poly__degree': 2,
        #   'transformer__poly__include_bias': False,
        #   'transformer__poly__interaction_only': False}

        # 'transformer__mpg_discretizer__n_bins': [5, 6, 10, 13],
        # 'transformer__mpg_discretizer__encode': ['onehot', 'ordinal'],
        # 'transformer__mpg_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
        #   {'transformer__mpg_discretizer__encode': 'onehot',
        #   'transformer__mpg_discretizer__n_bins': 6,
        #   'transformer__mpg_discretizer__strategy': 'uniform'}

        # 'transformer__tax_discretizer__n_bins': [7, 8, 9, 10],
        # 'transformer__tax_discretizer__encode': ['onehot', 'ordinal'],
        # 'transformer__tax_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
        #   {'transformer__tax_discretizer__encode': 'onehot',
        #   'transformer__tax_discretizer__n_bins': 9, '
        #   transformer__tax_discretizer__strategy': 'quantile'}

        # 'transformer__engine_size_discretizer__n_bins': [2, 3, 4, 6, 9],
        # 'transformer__engine_size_discretizer__encode': ['onehot', 'ordinal'],
        # 'transformer__engine_size_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
        #   {'transformer__engine_size_discretizer__encode': 'onehot',
        #   'transformer__engine_size_discretizer__n_bins': 3,
        #   'transformer__engine_size_discretizer__strategy': 'uniform'}

        # 'transformer__year_pipe__discretize__n_bins': [3, 6, 9, 11],
        # 'transformer__year_pipe__discretize__encode': ['onehot', 'ordinal'],
        # 'transformer__year_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans']
        #   {'transformer__year_pipe__discretize__encode': 'onehot',
        #   'transformer__year_pipe__discretize__n_bins': 11,
        #   'transformer__year_pipe__discretize__strategy': 'uniform'}
    }
    # *** GridSearchCV ***#
    # mse = make_scorer(mean_squared_error, greater_is_better=False)
    # model = get_best_estimator(
    #     model, param_grid, X_train, y_train, scoring=mse)
    # dump(model, model_path)

    # Sanity check
    # print('X_train',X_train.isna().any())
    # print('y_train',y_train.isna().any())