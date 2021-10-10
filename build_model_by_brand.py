from os.path import join, isfile
from os import listdir
from numpy.lib.utils import deprecate
import pandas as pd
import numpy as np
import json

from bs_lib.bs_eda import load_all_csv, load_csv_file
from bs_lib.bs_eda import train_val_test_split, get_ordered_categories
import prepare_data as prepare
import build_model as build
import optimize_model as optimizer
from joblib import load, dump

import constants as cnst

np.random.seed(1)

verbose = False

file_to_exclude = ['all_set.csv', 'all_brands.csv']
# current_directory = "."
# dataset_directory = "dataset"

# model_directory = "model"
# cnst.MODEL_DIR_PATH = join(current_directory, model_directory)


def get_model_params(categories):

    return {
        # "features_extraction": 'passthrough',
        "features_extraction__kw_args": {'features': ["age", "tax_per_year"]},
        # -----------
        # numerical
        # ___________
        # "transformer__poly": 'passthrough',
        # "transformer__mpg_pipe": 'passthrough',
        # "transformer__tax_pipe": 'passthrough',
        # "transformer__engine_size_pipe": 'passthrough',
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
        "transformer__fuel_type_pipe__OHE__categories": [categories['fuel_type']],
        "transformer__fuel_type_pipe__OE": 'passthrough',

        # "transformer__fuel_type_pipe__OHE": 'passthrough',
        # "transformer__fuel_type_pipe__OE__categories": [categories['fuel_type']],
    }


def get_data():
    all_df = load_all_csv(dataset_path=cnst.FILE_PROCESSED_PATH,
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


def get_set_split(data, target, verbose=False):
    df = data.copy()
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

        model_name = f'{brand}_model'
        model_path = build.dump_model(model, model_name, cnst.MODEL_DIR_PATH)
        if evaluate:
            report.append([brand.title(),
                           int(round(rmse, 0)),
                           int(round(y_val.mean(), 0)),
                           int(round(y_pred.mean(), 0)),
                           model_path])
    if evaluate:
        build.print_performance(report)


def get_best_models(verbose=False):
    all_df = get_data()

    for brand, dataframe in all_df.items():

        # get data
        filename = f"{brand}.csv"
        df = get_prepared_data(dataframe, filename)
        #categories = get_ordered_categories(data=df, by='price')
        _, X_val, _, _, y_val, _ = get_set_split(
            df, target='price')

        # load model
        model_name = f'model_{brand}.joblib'
        model = load(join(cnst.MODEL_DIR_PATH, model_name))

        best_model = model
        best_params = {}
        _, _, best_score = build.evaluate(model, X_val, y_val, verbose=verbose)
        for param in optimizer.get_combinations_of_params():
            model_, params_, _ = optimizer.evaluate_combinations(
                model, param, X_val, y_val, verbose=verbose)
            _, _, current_score = build.evaluate(
                model_, X_val, y_val, verbose=verbose)
            if verbose:
                print(
                    f"\nCurrent param combination Score: {current_score}, last best Score {best_score}")
            if current_score < best_score:
                best_model = model_
                best_params = params_
                best_score = current_score
                if verbose:
                    print("\nNew best solution:")
                    print("\nBest score", best_score)
                    print("\nBest params", best_params)

        # if verbose:
        print(f"\n{brand} - Best score", best_score)
        print(f"{brand} - Best params", best_params)
        model_name = f'{brand}_optimized_rmse_{best_score:.0f}'
        model_path = build.dump_model(
            best_model, model_name, cnst.MODEL_DIR_PATH)
        # if verbose:
        print(f'Best Model saved @ {model_path}')
        params_file_path = join(cnst.MODEL_DIR_PATH, f"{model_name}.json")
        with open(params_file_path, 'w') as file:
            json.dump(best_params, file)


def get_matching_files(in_directory_path, extension, search_term, exclude_term=[]):
    files = {}
    all_files_in_directory = listdir(in_directory_path)
    for filename in sorted(all_files_in_directory):
        # fetch model filename
        if (isfile(join(in_directory_path, filename))
                and filename.endswith(extension)
                and filename not in exclude_term
                and not any(x in filename for x in exclude_term)
                and search_term in filename):
            file_name = filename.split('_')[0]
            files[file_name] = filename
    return files


def evaluate_all_refitted_models(base_model_search_term, params_search_term, model_exclude=[], params_exclude=[], refit_times=1):
    report = []
    models_files = {}
    params_files = {}

    models_files = get_matching_files(cnst.MODEL_DIR_PATH, extension='.joblib',
                                      search_term=base_model_search_term, exclude_term=model_exclude)
    params_files = get_matching_files(
        cnst.MODEL_DIR_PATH, '.json', params_search_term, params_exclude)

    for brand_name, model_filename in models_files.items():
        print(
            f"Refit {model_filename} using {brand_name}.csv with parameters from {params_files[brand_name]}")
        # load dataset
        df = prepare.load_prepared_file(filename=f"{brand_name}.csv")
        X_train, X_val, _, y_train, y_val, _ = get_set_split(
            df, target='price')
        # load params
        model = build.load_model_with_params(
            model_filename=model_filename, params_filename=params_files[brand_name])
        n_estimator = 500
        if refit_times > 1:
            model.set_params(**{"random_forest__warm_start": True})
        # refit
            for i in range(refit_times):
                model.set_params(
                    **{"random_forest__n_estimators": (i+1)*n_estimator})
                model.fit(X_train, y_train)
        else:
            model.set_params(**{"random_forest__n_estimators": n_estimator})
            model.fit(X_train, y_train)
        # evaluate against Val Set
        y_pred, y_val, rmse = build.evaluate(model, X_val, y_val)
        error = np.sqrt(np.square(y_val-y_pred))
        #score = model.score(X_val,y_val)
        report.append([brand_name.title(),
                       int(round(rmse, 0)),
                       # score,
                       int(round(error.mean(), 0)),
                       model_filename.split('.')[0]])
    build.print_performance(
        report, ['Brand', 'RMSE', 'Absolute Mean Error', 'Model path'])


def evaluate_all_models(search_term, exclude=[]):
    report = []
    # fetch all models files
    models_files = get_matching_files(
        cnst.MODEL_DIR_PATH, extension='.joblib', search_term=search_term, exclude_term=exclude)

    for brand_name, model_filename in models_files.items():
        df = prepare.load_prepared_file(filename=f"{brand_name}.csv")
        X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(
            df, target='price')
        model_path = join(cnst.MODEL_DIR_PATH, model_filename)
        model = load(model_path)
        y_pred, y_val, rmse = build.evaluate(model, X_val, y_val)
        error = np.sqrt(np.square(y_val-y_pred))
        #score = (model.score(X_train,y_train)*0.75+model.score(X_val,y_val)*0.15+model.score(X_test,y_test)*0.1)
        report.append([brand_name.title(),
                       int(round(rmse, 0)),
                       # score,
                       int(round(error.mean(), 0)),
                       model_filename.split('.')[0]])
    build.print_performance(
        report, ['Brand', 'RMSE', 'Absolute Mean Error', 'Model path'])


def get_best_estimators_params(search_term, exclude_term=[], verbose=False, prefix=''):

    all_df = get_data()
    models_files = get_matching_files(
        cnst.MODEL_DIR_PATH, extension='.joblib', search_term=search_term, exclude_term=exclude_term)

    for brand, dataframe in all_df.items():
        # get data
        filename = f"{brand}.csv"
        df = get_prepared_data(dataframe, filename)

        X_train, X_val, _, y_train, y_val, _ = get_set_split(df, target='price')

        # load model
        model = load(join(cnst.MODEL_DIR_PATH, models_files[brand]))

        # get reference score
        _, _, ref_score = build.evaluate(model, X_train, y_train, verbose=verbose)

        # Search for best params value
        best_model, best_params, _ = optimizer.evaluate_combination(
            model, optimizer.estimator_params, X_train, y_train, verbose=verbose)

        # Reevaluate with best_model
        _, _, best_score = build.evaluate(
            best_model, X_train, y_train, verbose=verbose)
        if len(prefix) > 0:
            _prefix = f'{prefix}-'
        else:
            _prefix = ''

        output_model_name = f'{brand}_{_prefix}estimator'
        model_to_dump = model
        params_to_save = {}

        if(best_score <= ref_score):
            if verbose:
                print(f"\n{brand}")
                print(f"-\t Reference score: {ref_score}")
                print(f"-\t Best score: {best_score}")
                print(f"-\t Best params: {best_params}")

            output_model_name = f"{output_model_name}_{best_score:.0f}"
            model_to_dump = best_model

            # encode int64 to int
            encoded_best_params = {}
            for key, p in best_params.items():
                for k, value in p.items():
                    print(f"{k}:{value}")
                    if isinstance(value, np.int64):
                        encoded_best_params[k] = int(value)
                    elif isinstance(value, np.float64):
                        encoded_best_params[k] = float(value)
                    else:
                        encoded_best_params[k] = value

            params_to_save = encoded_best_params
        else:
            if verbose:
                print(f"\n{brand}")
                print(f"-\t Reference score: {ref_score}")
                print(f"-\t Optimisation Estimator score: {best_score}")

        model_path = build.dump_model(
            model_to_dump, output_model_name, cnst.MODEL_DIR_PATH)

        if verbose:
            print(f'-\t Estimator optimised Model saved @ {model_path}')

        params_file_path = join(cnst.MODEL_DIR_PATH,
                                f"{output_model_name}.json")
        with open(params_file_path, 'w') as file:
            json.dump(params_to_save, file)


if __name__ == "__main__":

    # get_model()
    get_best_models(verbose=False)
    # evaluate_all_models()
