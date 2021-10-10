from os.path import join, isfile
import pandas as pd
import numpy as np

from bs_lib.bs_eda import load_all_csv, load_csv_file
from bs_lib.bs_eda import train_val_test_split, get_ordered_categories
import prepare_data as prepare
import build_model as build
import constants as cnst
import optimize_model as optimizer

from joblib import load, dump
import json

def get_params(data):
    categories = get_ordered_categories(data=data, by='price')
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
    return params


def get_model(model_filename, data_filename, verbose=False):
    model_file_path = join(cnst.MODEL_DIR_PATH, model_filename)
    if isfile(model_file_path):  # if model already exist
        print(f"Loading {model_filename}")
        model = load(model_file_path)
    else:
        print(f'{model_filename} not found')
        transformers = build.get_transformer(verbose=verbose)
        model = build.get_model(transformers=transformers, verbose=verbose)
        data = get_data(data_filename, verbose=verbose)

        params = get_params(data)

        X_train, _, _, y_train, _, _ = get_split_sets(data)

        # print(model.get_params().keys())
        model.set_params(**params)
        model.fit(X_train, y_train)

        build.dump_model(model, model_filename.split('.')[0], verbose=verbose)

    return model


def get_processed_file(data_filename, verbose=False):
    file_path = join(cnst.FILE_PROCESSED_PATH, data_filename)
    if isfile(file_path):
        if verbose:
            print(f'Loading {data_filename} from {cnst.FILE_PROCESSED_PATH}')
        df = load_csv_file(file_path, index=0)
    else:
        if verbose:
            print(f'{data_filename} not found in {cnst.FILE_PROCESSED_PATH}')
        all_df_dict = load_all_csv(
            dataset_path=cnst.FILE_PROCESSED_PATH, exclude=[data_filename], index=0, verbose=verbose)

        columns = all_df_dict['audi'].columns.to_list()
        # load all files
        df = pd.DataFrame(columns=columns, dtype=float)
        # merge all files
        for brand, dataframe in all_df_dict.items():
            if verbose:
                print('Dataframe with all files merged:', dataframe.info())
            df = pd.concat([df, dataframe], ignore_index=True)
        # print(df.info())
        df.to_csv(file_path)
    return df


def get_data(data_filename, verbose=False):
    prepared_df = prepare.load_prepared_file(filename=data_filename)
    if isinstance(prepared_df, pd.DataFrame):
        df = prepared_df
        if verbose:
            print(f'Prepared {data_filename} found:', df.info())

    else:
        df = get_processed_file(data_filename, verbose=verbose)
        print(f'Before preparing data: {df.info()}')
        df = prepare.clean_variables(df)
        df = prepare.nan_outliers(df)
        df = prepare.numerical_imputer(
            df, n_neighbors=10, weights='distance', imputer_type='KNN')

        prepare.save_prepared_file(df, filename=data_filename)
    return df


def get_split_sets(df, verbose=False):
    data = df.copy()
    target = 'price'
    data_target = data[target]
    data.drop(target, axis=1, inplace=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=data,
                                                                          y=data_target,
                                                                          train_size=.75,
                                                                          val_size=.15,
                                                                          test_size=.1,
                                                                          random_state=1,
                                                                          show=verbose)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model_filename, data_filename, verbose=False):
    # get the model
    model = get_model(model_filename, data_filename, verbose=verbose)
    data = get_data(data_filename, verbose=verbose)
    _, X_val, _, _, y_val, _ = get_split_sets(data, verbose=verbose)
    y_pred, y_val, rmse = build.evaluate(model, X_val, y_val, verbose=verbose)
    error = np.sqrt(np.square(y_val-y_pred))
    #score = model.score(X_val,y_val)
    model_name = model_filename.split('.')[0]
    model_name = model_name[:20] + (model_name[20:] and '...')
    report = [[f"Merged Brand Dataset",
                       int(round(rmse, 0)),
                       #score,
                       int(round(error.mean(), 0)),
                       model_name]]
    build.print_performance(report, ['Brand', 'RMSE', 'Absolute Mean Error', 'Model path'])


def get_best_estimator(model_filename, data_filename, verbose=False):
    model_name = model_filename.split('.')[0]
    model = get_model(model_filename, data_filename, verbose=verbose)
    data = get_data(data_filename, verbose=verbose)
    _, X_val, _, _, y_val, _ = get_split_sets(data, verbose=verbose)
    _, _, ref_score = build.evaluate(model, X_val, y_val, verbose=verbose)
    best_model, best_params, _ = optimizer.evaluate_combination(
        model, optimizer.estimator_params, X_val, y_val, verbose=verbose)
    _, _, best_score = build.evaluate(
        best_model, X_val, y_val, verbose=verbose)
    if(best_score <= ref_score):
        if verbose:
            print(f"\n{model_name}")
            print(f"-\t Reference score: {ref_score}")
            print(f"-\t Best score: {best_score}")
            print(f"-\t Best params: {best_params}")
        model_name = f'{model_name}_estimator_optimized_rmse_{best_score:.0f}'
        model_path = build.dump_model(
            best_model, model_name, cnst.MODEL_DIR_PATH)

        if verbose:
            print(f'-\t Best Model saved @ {model_path}')

        # encode int64 to int
        encoded_best_params = {}
        for key, p in best_params.items():
            for k, value in p.items():
                if verbose:
                    print(f"{k}:{value}")
            if isinstance(value, np.int64):
                encoded_best_params[key] = int(value)
            elif isinstance(value, np.float64):
                encoded_best_params[key] = float(value)
            else:
                encoded_best_params[key] = value
        params_file_path = join(cnst.MODEL_DIR_PATH, f"{model_name}.json")
        with open(params_file_path, 'w') as file:
            json.dump(encoded_best_params, file)
    else:
        if verbose:
            print(f"\n{model_name}")
            print(f"-\t Reference score: {ref_score}")
            print(f"-\t Optimisation Estimator score: {best_score}")
        model_name = f'{model_name}_estimator_optimized_rmse_{ref_score:.0f}'
        model_path = build.dump_model(
            model, model_name, cnst.MODEL_DIR_PATH)

        if verbose:
            print(f'-\t Model saved @ {model_path}')


if __name__ == "__main__":

    np.random.seed(1)

    verbose = False
    filename = "all_set.csv"
    file_path = join(cnst.FILE_PROCESSED_PATH, filename)

    # ------------------------------------------------------------------------#
    # Load files processed dataset
    # ________________________________________________________________________#

    # if file doesn't already exist
    if isfile(file_path):
        all_set_df = load_csv_file(file_path, index=0)
    else:
        all_df_dict = load_all_csv(
            dataset_path=cnst.ORIGINAL_DATA_PATH, exclude=[], index=0)

        columns = all_df_dict['audi'].columns.to_list()
        # load all files
        all_set_df = pd.DataFrame(columns=columns, dtype=float)
        # merge all files
        for brand, dataframe in all_df_dict.items():
            # print(dataframe.info())
            all_set_df = pd.concat([all_set_df, dataframe], ignore_index=True)
        # print(all_set_df.info())
        all_set_df.to_csv(file_path)

    # ------------------------------------------------------------------------#
    # prepare dataset
    # ________________________________________________________________________#

    prepared_df = prepare.load_prepared_file(filename=filename)
    if isinstance(prepared_df, pd.DataFrame):
        all_set_df = prepared_df
    else:

        all_set_df = prepare.clean_variables(all_set_df)
        all_set_df = prepare.nan_outliers(all_set_df)
        all_set_df = prepare.numerical_imputer(
            all_set_df, n_neighbors=10, weights='distance', imputer_type='KNN')

        prepare.save_prepared_file(all_set_df, filename=filename)

    # ------------------------------------------------------------------------#
    # split dataset
    # ________________________________________________________________________#

    target = 'price'
    categories = get_ordered_categories(data=all_set_df, by=target)

    df_target = all_set_df[target]
    all_set_df.drop(target, axis=1, inplace=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=all_set_df,
                                                                          y=df_target,
                                                                          train_size=.75,
                                                                          val_size=.15,
                                                                          test_size=.1,
                                                                          random_state=1,
                                                                          show=verbose)

    # ------------------------------------------------------------------------#
    # build model
    # ________________________________________________________________________#
    model = get_model()
    model.fit(X_train, y_train)
    y_pred, y_val, rmse = build.evaluate(model, X_val, y_val)

    build.print_performance([[f"Merged Brand Dataset",
                              int(round(rmse, 0)),
                              int(round(y_val.mean(), 0)),
                              int(round(y_pred.mean(), 0)),
                              '-']])
