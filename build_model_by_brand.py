from os.path import join, isfile
from os import listdir
from numpy.lib.utils import deprecate
import pandas as pd
import numpy as np

from bs_lib.bs_eda import load_all_csv, load_csv_file
from bs_lib.bs_eda import train_val_test_split, get_ordered_categories
import prepare_data as prepare
import build_model as build
import optimize_model as optimizer
from joblib import load, dump

np.random.seed(1)

verbose = False

file_to_exclude = ['all_set.csv']
current_directory = "."
dataset_directory = "dataset"

model_directory = "model"
model_directory_path = join(current_directory, model_directory)


def get_model_params(categories):

    return {
        "features_extraction":'passthrough',
        #"features_extraction__kw_args": {'features': ["age", "tax_per_year"]},
        # -----------
        # numerical
        # ___________
        "transformer__poly": 'passthrough',
        "transformer__mpg_pipe": 'passthrough',
        "transformer__tax_pipe": 'passthrough',
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


def get_data():
    prefix = 'file_processed'
    file_directory = join(current_directory, dataset_directory, prefix)
    all_df = load_all_csv(dataset_path=file_directory,
                          exclude=file_to_exclude, verbose=False)
    return all_df


def get_prepared_data(df, filename):
    _df = prepare.load_prepared_file(filename=filename)
    if not isinstance(_df, pd.DataFrame):
        _df = df.copy()
        _df = prepare.clean_variables(_df)
        _df = prepare.nan_outliers(_df)
        _df = prepare.numerical_imputer(
            _df, n_neighbors=10, weights='distance', imputer_type='KNN')

        prepare.save_prepared_file(_df, filename=filename)
    return _df


def get_set_split(df, target):
    df_target = df[target]
    df.drop(target, axis=1, inplace=True)
    return train_val_test_split(X=df,
                                y=df_target,
                                train_size=.75,
                                val_size=.15,
                                test_size=.1,
                                random_state=1,
                                show=verbose)


def get_model(evaluate=True):

    # Load files processed dataset
    # files already processed using file_processing.py
    all_df = get_data()

    target = 'price'

    if evaluate:
        report = []

    for brand, dataframe in all_df.items():
        filename = f"{brand}.csv"
        # prepare dataset
        df = get_prepared_data(dataframe, filename)
        categories = get_ordered_categories(data=df, by=target)

        # split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(
            df, target=target)

        # build model
        transformers = build.get_transformer(verbose=False)
        model = build.get_model(transformers=transformers, verbose=False)
        params = get_model_params(categories)

        model.set_params(**params)

        model.fit(X_train, y_train)

        y_pred, y_val, rmse = build.evaluate(model, X_val, y_val)

        model_name = f'model_{brand}'
        model_path = build.dump_model(model, model_name, model_directory_path)
        if evaluate:
            report.append([brand.title(),
                           int(round(rmse, 0)),
                           int(round(y_val.mean(), 0)),
                           int(round(y_pred.mean(), 0)),
                           model_path])
    if evaluate:
        build.print_performance(report)

# def optimize():

#     # exclude = []
#     # models = [join(model_directory_path, f) for f in listdir(model_directory_path) if (
#     #     isfile(join(model_directory_path, f)) and f.endswith('.joblib') and f not in exclude)]
#     # print(f'loading:{models}')

#     all_df = get_data()

#     for brand, dataframe in all_df.items():
#         # get data
#         filename = f"{brand}.csv"
#         df = get_prepared_data(dataframe, filename)
#         categories = get_ordered_categories(data=df, by='price')
#         X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(
#             df, target='price')

#         # load model
#         model_name = f'model_{brand}.joblib'
#         model = load(join(model_directory_path, model_name))
#         # print(model.get_params(['transformer']))
#         optimizer.optimize(model, X_train, y_train)


def get_best_models():
    all_df = get_data()

    for brand, dataframe in all_df.items():

        # get data
        filename = f"{brand}.csv"
        df = get_prepared_data(dataframe, filename)
        #categories = get_ordered_categories(data=df, by='price')
        X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(
            df, target='price')

        # load model
        model_name = f'model_{brand}.joblib'
        model = load(join(model_directory_path, model_name))

        best_model = model
        best_params = {}
        _, _, best_score = build.evaluate(model, X_val, y_val, verbose=True)

        for param in optimizer.get_combinations_of_params():
            model_, params_, _ = optimizer.evaluate_combination(
                model, param, X_val, y_val, verbose=True)
            _, _, current_score = build.evaluate(
                model_, X_val, y_val, verbose=False)
            print(
                f"\nCurrent param combination Score: {current_score}, last best Score {best_score}")
            if current_score < best_score:
                best_model = model_
                best_params = params_
                best_score = current_score
                print("\nNew best solution:")
                print("\nBest score", best_score)
                print("\nBest params", best_params)

        print(f"\n{brand} - Best score", best_score)
        print(f"{brand} - Best params", best_params)
        model_name = f'{brand}_optimize_rmse_{best_score:.0f}'
        model_path = build.dump_model(
            best_model, model_name, model_directory_path)
        print(f'Best Model saved @ {model_path}')


def evaluate_all_models():
    report = []
    exclude = []
    # load all models
    models = []
    for f in listdir(model_directory_path):
        if (isfile(join(model_directory_path, f))
                and f.endswith('.joblib')
                and f not in exclude
                and 'optimize' in f):
            brand = f.split('_')[0]
            filename = f"{brand}.csv"
            df = prepare.load_prepared_file(filename=filename)
            X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(df, target='price')
            model_path = join(model_directory_path, f)
            model = load(model_path)
            y_pred, y_val, rmse = build.evaluate(model, X_val, y_val)
            report.append([brand.title(),
                        int(round(rmse, 0)),
                        int(round(y_val.mean(), 0)),
                        int(round(y_pred.mean(), 0)),
                        model_path])
    build.print_performance(report)

if __name__ == "__main__":
    get_model()
    #get_best_models()
    #evaluate_all_models()
