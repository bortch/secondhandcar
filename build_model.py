# coding=utf-8
import json
from joblib import dump, load
from bs_lib.bs_eda import train_val_test_split, get_ordered_categories
from bs_lib.bs_eda import load_csv, load_all_csv
import bs_lib.bs_preprocess_lib as bsp
import bs_lib.bs_transformer as tsf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import bs_lib.bs_terminal as terminal
import pandas as pd
import numpy as np
from matplotlib.pyplot import title
from os.path import join, isfile
from os import rename
import sys
from bs_lib.bs_eda import get_categorical_columns
from pandas.core.indexes import category
import warnings
warnings.filterwarnings('ignore')
import constants as cnst

def get_transformer(verbose=False):
    if verbose:
        print("\nCreating Columns transformers")

    fuel_type_pipeline = Pipeline(steps=[
        ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        ('OE', OrdinalEncoder()),
    ], verbose=verbose)

    transformers_ = [
        ("poly", PolynomialFeatures(degree=2,
                                    interaction_only=False,
                                    include_bias=False),
         make_column_selector(dtype_include=np.number)),

        ("mpg_pipe", Pipeline(steps=[
            ('discretize', KBinsDiscretizer(n_bins=6,
                                            encode='onehot', strategy='uniform'))
        ], verbose=verbose), ['mpg']),

        ("tax_pipe", Pipeline(steps=[
            ('discretize',  KBinsDiscretizer(n_bins=9,
                                             encode='onehot', strategy='quantile'))
        ], verbose=verbose), ['tax']),

        ("engine_size_pipe", Pipeline(steps=[
            ('discretize',  KBinsDiscretizer(n_bins=3,
                                             encode='onehot', strategy='uniform'))
        ], verbose=verbose), ['engine_size']),

        ('year_pipe', Pipeline(steps=[
            ('discretize', KBinsDiscretizer(
                n_bins=10, encode='ordinal', strategy='quantile'))
        ], verbose=verbose), ['year']),

        ('model_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['model']),

        ('brand_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['brand']),

        ('transmission_pipe', Pipeline(steps=[
            ('OHE', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ('OE', OrdinalEncoder())
        ], verbose=verbose), ['transmission']),

        ('fuel_type_pipe', fuel_type_pipeline, ['fuel_type']),
    ]

    transformer = ColumnTransformer(
        transformers_, remainder='passthrough', verbose=verbose)
    return transformer


def extract_features(data, features=['all']):
    X = data.copy()
    if 'all' in features:
        features = ['model_count', 'age',
                    'mpy_mpy', 'tax_per_year',
                    'mileage_per_year', 'mpg_per_year',
                    'engine_per_year']
    # drop testing
    # ['year', 'price', 'mileage', 'tax', 'mpg', 'engine_size']
    # X.drop(['year'],axis=1,inplace=True)
    # X.drop(['tax'],axis=1,inplace=True)
    # X.drop(['mileage'],axis=1,inplace=True)
    # X.drop(['mpg'],axis=1,inplace=True)
    # X.drop(['engine_size'],axis=1,inplace=True)
    # X.drop(['brand'],axis=1,inplace=True)

    # adding feature
    model_count = X.groupby('model')['model'].transform('count')
    occ = model_count/X.shape[0]
    if 'model_count' in features:
        X['model_count'] = model_count

    age = X['year'].max()-X['year']
    age[age < 1] = 1
    if 'age' in features:
        X['age'] = age
        #X.loc[X['age'] < 1, 'age'] = 1

    m_a = X['mileage']/age
    if 'mileage_per_year' in features:
        X['mileage_per_year'] = m_a

    mpg_a = X['mpg']/age
    if 'mpg_per_year' in features:
        X['mpg_per_year'] = mpg_a

    t_a = X['tax']/age
    if 'tax_per_year' in features:
        X['tax_per_year'] = t_a

    e_a = X['engine_size']/age
    if 'engine_per_year' in features:
        X['engine_per_year'] = e_a

    mmte = (X['mileage']+X['mpg']+X['tax']+X['engine_size'])/occ
    if 'mpy_mpy' in features:
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


def evaluate(model, X_val, y_val, verbose=False):
    if verbose:
        print(f"\nModel Evaluation")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    if verbose:
        print(f"RMSE: {rmse}")
    return np.exp(y_pred), np.exp(y_val), rmse


def evaluate_prediction(model, X_val, y_val, sample=None,add_columns=[]):
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
        sample = X_val[i:i+1]
        pred = y_pred[i:i+1][0]
        real = int(y_val[i:i+1].values[0])
        error = (real-pred)
        percentage = (error/real)*100
        row.append(f"{pred:.0f}")
        row.append(f"{real:.0f}")
        row.append(f"{error:.0f}")
        row.append(f"{percentage:.0f} %")
        for col in add_columns:
            value = sample[col].array[0]
            row.append(f"{value}")
        data.append(row)

    table_columns = ['Prediction', 'Real Price', 'Error', 'Percentage']+add_columns 
    table = terminal.create_table(title="Prediction results",
                                  columns=table_columns,
                                  data=data)
    terminal.article(title="Model Prediction testing", content=table)


def print_performance(data, columns=[]):
    if len(columns)==0:
        columns=['Brand', 'RMSE', 'Target mean', 'Prediction mean', 'Model path']
    table = terminal.create_table(title="One model by Brand",
                                  columns=columns, data=data)
    terminal.article(title="Performance Comparison", content=table)


def fit_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    evaluate(model, X_val, y_val)


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


def dump_model(model, as_filename, verbose=False):
    model_filename = f'{as_filename}.joblib'
    dumped_model_path = join(cnst.MODEL_DIR_PATH, model_filename)
    dump(model, dumped_model_path)
    if verbose:
        print(f"Model {as_filename} saved @ {dumped_model_path}")
    return dumped_model_path


def get_model(model_path_to_load=None, verbose=False, warm_start=False, transformers=None):

    if (model_path_to_load is not None) and isfile(model_path_to_load):
        model = load(model_path_to_load)
        return model
    else:
        if transformers == None:
            transformers_ = get_transformer(verbose=verbose)
        else:
            transformers_ = transformers

        nb_estimators = 10
        steps = [
            ("features_extraction", FunctionTransformer(
                extract_features, validate=False)),
            ("transformer", transformers_),
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
        pipeline = Pipeline(steps=steps, verbose=verbose)
        return pipeline


def get_all_base_models(files_directory, target, model_dump=False,verbose=False):

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
        params = {
            # "features_extraction":'passthrough',
            "features_extraction__kw_args": {'features': ["age", "tax_per_year"]},
            # -----------
            # numerical
            # ___________
            "transformer__poly": 'passthrough',
            "transformer__mpg_pipe": 'passthrough',
            # "transformer__tax_pipe": 'passthrough',
            "transformer__engine_size_pipe": 'passthrough',
            "transformer__year_pipe": 'passthrough',
            # -------------
            # categorical
            # -------------
            # *** model
            # "transformer__model_pipe__OHE__categories": [categories['model']],
            # "transformer__model_pipe__OE": 'passthrough',

            "transformer__model_pipe__OHE": 'passthrough',
            "transformer__model_pipe__OE__categories": [categories['model']],

            # *** brand
            # "transformer__brand_pipe__OHE__categories": [categories['brand']],
            # "transformer__brand_pipe__OE": 'passthrough',

            "transformer__brand_pipe__OHE": 'passthrough',
            "transformer__brand_pipe__OE__categories": [categories['brand']],

            # *** transmission
            # "transformer__transmission_pipe__OHE__categories": [categories['transmission']],
            # "transformer__transmission_pipe__OE": 'passthrough',

            "transformer__transmission_pipe__OHE": 'passthrough',
            "transformer__transmission_pipe__OE__categories": [categories['transmission']],

            # *** fuel_type
            "transformer__fuel_type_pipe__OHE__categories": [categories['fuel_type']],
            "transformer__fuel_type_pipe__OE": 'passthrough',

            # "transformer__fuel_type_pipe__OHE": 'passthrough',
            # "transformer__fuel_type_pipe__OE__categories": [categories['fuel_type']],
        }

        # print(model.get_params().keys())
        model.set_params(**params)

        print(f"\nTraining the model for {brand}")
        model.fit(X_train, y_train)
        model_path = '-'
        # if model_dump:
        #     temp_model_name = f'temp_{brand}_model'
        #     temp_model_filename = f'{temp_model_name}.joblib'
        #     temp_model_path = join(cnst.MODEL_DIR_PATH, temp_model_filename)
        #     dump(model, temp_model_path)
        #     print(f"Model {brand}:\n")

        y_pred, y_val, rmse = evaluate(model, X_val, y_val)

        if model_dump:
            model_name = f'model_{brand}_{round(rmse,0)}'
            # model_filename = f'{model_name}.joblib'
            # model_path = join(cnst.MODEL_DIR_PATH, model_filename)
            # rename(temp_model_path, model_path)
            model_path = dump_model(model, model_name)

        report.append([brand.title(),
                       int(round(rmse, 0)),
                       int(round(y_val.mean(), 0)),
                       int(round(y_pred.mean(), 0)),
                       model_path])

    return report


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
        results[cat] = []
        for c in df[cat].values:
            results[cat].append(c)
        # print(results[cat])
    return results


def get_one_model_for_all_iterative(model_to_dump=False, evaluation=False, verbose=False):
    
    all_df = load_all_csv(dataset_path=cnst.PREPARED_DATA_PATH, index=0,exclude=["all_set.csv"])
    categories = get_all_categories(all_df)
    report = []

    model = get_model(warm_start=True, verbose=False)
    params = {
        # "features_extraction":'passthrough',
        "features_extraction__kw_args": {'features': [
            'model_count',
            # 'age',
            'mpy_mpy',
            'tax_per_year',
            # 'mileage_per_year',
            # 'mpg_per_year',
            # 'engine_per_year'
        ]},
        # -----------
        # numerical
        # ___________
        "transformer__poly": 'passthrough',
        # "transformer__mpg_pipe": 'passthrough',
        "transformer__tax_pipe": 'passthrough',
        "transformer__engine_size_pipe": 'passthrough',
        # "transformer__year_pipe": 'passthrough',
        # -------------
        # categorical
        # -------------
        # *** model
        # "transformer__model_pipe__OHE__categories": [categories['model']],
        # "transformer__model_pipe__OE": 'passthrough',

        "transformer__model_pipe__OHE": 'passthrough',
        "transformer__model_pipe__OE__categories": [categories['model']],

        # *** brand
        # "transformer__brand_pipe__OHE__categories": [categories['brand']],
        # "transformer__brand_pipe__OE": 'passthrough',

        "transformer__brand_pipe__OHE": 'passthrough',
        "transformer__brand_pipe__OE__categories": [categories['brand']],

        # *** transmission
        # "transformer__transmission_pipe__OHE__categories": [categories['transmission']],
        # "transformer__transmission_pipe__OE": 'passthrough',

        "transformer__transmission_pipe__OHE": 'passthrough',
        "transformer__transmission_pipe__OE__categories": [categories['transmission']],

        # *** fuel_type
        # "transformer__fuel_type_pipe__OHE__categories": [categories['fuel_type']],
        # "transformer__fuel_type_pipe__OE": 'passthrough',

        "transformer__fuel_type_pipe__OHE": 'passthrough',
        "transformer__fuel_type_pipe__OE__categories": [categories['fuel_type']],
    }
    model.set_params(**params)
    n_estimators = 10
    columns = ['model', 'year', 'transmission',
               'mileage', 'fuel_type',
               'tax', 'mpg',
               'engine_size', 'brand']
    all_X_val = pd.DataFrame(columns=columns, dtype=float)
    all_y_val = pd.Series()
    i = 1
    # print(f"\nTraining the model...")
    target = 'price'
    for brand, df in all_df.items():
        df_target = df[target]
        df = df.drop(target, axis=1)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=df,
                                                                              y=df_target,
                                                                              train_size=.75,
                                                                              val_size=.15,
                                                                              test_size=.1,
                                                                              random_state=1,
                                                                              show=verbose)
        estimators = i*n_estimators
        model.set_params(**{"random_forest__n_estimators": estimators})
        model.fit(X_train, y_train)

        all_X_val = all_X_val.append(X_val)
        all_y_val = all_y_val.append(y_val)
        i += 1

    y_pred, y_val, rmse = evaluate(model, all_X_val, all_y_val)

    if model_to_dump:
        model_name = f'all_brand_model_{round(rmse,0)}'
        model_path = dump_model(model, model_name, verbose=False)
    
    error = np.sqrt(np.square(y_val-y_pred))
    score = model.score(all_X_val,all_y_val)
    report.append([f"Inc. Model",
                       int(round(rmse, 0)),
                       score,
                       int(round(error.mean(), 0)),
                       model_name])
    if evaluation:
        print_performance(report, ['Brand', 'RMSE', 'R2', 'Absolute Mean Error', 'Model path'])
    
    return model

def load_model_with_params(model_filename,params_filename):
    model_path = join(cnst.MODEL_DIR_PATH, model_filename)
    model = load(model_path)
    params_path = join(cnst.MODEL_DIR_PATH, params_filename)
    with open(params_path, 'r') as params_file:
        params = json.load(params_file)
    #print(params)
    clean_params = {}
    for key, param in params.items():
        if isinstance(param,dict):
            clean_params = {**clean_params,**param}
        else:
            clean_params[key]=param
    #print(clean_params)
    return model.set_params(**clean_params)


if __name__ == "__main__":
    np.random.seed(1)

    pd.options.mode.use_inf_as_na = True

    # all_data_file = 'all_set.csv'
    # train_set_file = 'train_set.csv'
    # val_set_file = 'val_set.csv'

    evaluations = get_all_base_models(
        cnst.PREPARED_DATA_PATH, target='price', verbose=False)
    print_performance(evaluations)

    # ofa_evaluations = get_one_model_for_all_iterative(
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
