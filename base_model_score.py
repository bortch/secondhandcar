from os.path import join, isfile
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

import bs_lib.bs_transformer as tsf
import bs_lib.bs_preprocess_lib as bsp

from joblib import dump, load


def categorize(X):
    X = pd.DataFrame(X).copy()
    columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"categorize {columns}")
    for c in columns:
        X[c] = X[c].astype('category')
        X[c] = X[c].cat.codes
    return X


def get_transformer(X):

    cat_columns = X.select_dtypes(include=['object']).columns.tolist()

    transformer = make_column_transformer(
        (tsf.TypeConverter('float'), ['mileage']),
        (OneHotEncoder(handle_unknown='ignore'), cat_columns),
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        remainder='passthrough', verbose=2)
    return transformer


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(rmse)


def fit_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    evaluate(model, X_val, y_val)

def get_data(file_name, target_column, dataset_directory_path='./dataset/', sample=None, callback=None, prefix='saved', show=False):
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
            print(f"{file_name} loaded\n A new file was saved as {saved_file_name}")
    if sample:
        data = data.sample(n=sample)
    X = data.drop(labels=[target_column], axis=1)
    y = np.log(data[target_column])  # plus normalisation
    if show:
        print(f"X: {X.shape}\ny: {y.shape}")
    return X, y

def clean_variables(data):
    df = data.copy()
    # remove duplicate
    df.drop_duplicates(inplace=True)
    # remove unhandled categories
    #df = df[df['transmission'] != 'Other']
    #df = df[(df['fuel_type'] != 'Other')]
    #df = df[(df['fuel_type'] != 'Electric')]
    return df

def prepare_data(data):
    df = data.copy()
    # columns = get_numerical_columns(df)
    # thresh = 3
    # outliers = df[columns].apply(lambda x: np.abs(
    #     zscore(x, nan_policy='omit')) > thresh)
    # # replace value from outliers by nan
    # df[outliers] = np.nan

    # imputer = KNNImputer(weights='distance')
    #  #columns = get_numerical_columns(df)
    # df[columns] = pd.DataFrame(imputer.fit_transform(df[columns]), columns=columns)
    return df

if __name__ == "__main__":
    np.random.seed(1)

    dataset_directory_path = 'dataset/'
    model_directory_path = 'model/'
    train_set_file = 'train_set.csv'
    val_set_file = 'val_set.csv'

    X_train, y_train = get_data(file_name=train_set_file, target_column='price',
                                dataset_directory_path=dataset_directory_path,
                                sample=None,
                                callback=[clean_variables,
                                          prepare_data],
                                show=True,prefix='base')
    X_val, y_val = get_data(file_name=val_set_file, target_column='price',
                            dataset_directory_path=dataset_directory_path,
                            sample=None, callback=[clean_variables],
                            show=True,prefix='base')

    transformer = get_transformer(X_train)

    # testing
    # X_ = pd.DataFrame(transformer.fit_transform(X).toarray())
    # print(X_.info())
    
    # If model not already exists:
    model_filename = 'base_score_model.joblib'
    model_path = join(model_directory_path, model_filename)
    if isfile(model_path):
        model = load(model_path)
        print(model)
    else:
        # pipeline: predict preprocessing
        model = make_pipeline(transformer,
                              RandomForestRegressor(
                                  n_estimators=100, verbose=1,n_jobs=-1),
                              verbose=True)
        model.fit(X_train, y_train)
        dump(model, model_path)

    evaluate(model, X_val, y_val)

    # score: 1874.8402339000947
 
    # bsp.get_learning_curve(model, X_train, y_train,
    #                        scoring='neg_mean_squared_error')
