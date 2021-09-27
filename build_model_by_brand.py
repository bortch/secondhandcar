from os.path import join, isfile
from os import listdir
import pandas as pd
import numpy as np

from bs_lib.bs_eda import load_all_csv, load_csv_file
from bs_lib.bs_eda import train_val_test_split, get_ordered_categories
import prepare_data as prepare
import build_model as build
import optimize_model as optimizer
from joblib import load

np.random.seed(1)

verbose = False

file_to_exclude = ['all_set.csv']
current_directory = "."
dataset_directory = "dataset"

model_directory = "model"
model_directory_path = join(current_directory, model_directory)


def get_model_params(categories):

    return {
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


def get_set_split(df,target):
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
        X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(df, target=target)
        
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


def optimize():

    # exclude = []
    # models = [join(model_directory_path, f) for f in listdir(model_directory_path) if (
    #     isfile(join(model_directory_path, f)) and f.endswith('.joblib') and f not in exclude)]
    # print(f'loading:{models}')
    
    all_df = get_data()

    for brand, dataframe in all_df.items():
        filename = f"{brand}.csv"
        df = get_prepared_data(dataframe, filename)
        categories = get_ordered_categories(data=df, by='price')  
        X_train, X_val, X_test, y_train, y_val, y_test = get_set_split(df,target='price')
        
        #load model
        model_name = f'model_{brand}.joblib'
        model = load(join(model_directory_path,model_name))
        optimizer.optimize(model, X_train, y_train)

if __name__ == "__main__":
    #get_model(evaluate=True)
    optimize()
