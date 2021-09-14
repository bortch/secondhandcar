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


if __name__ == "__main__":
    np.random.seed(1)

    dataset_directory_path = 'dataset/'
    model_directory_path = 'model/'
    dataset_file = 'train_set.csv'
    data = pd.read_csv(join(dataset_directory_path, dataset_file), index_col=0)
    # training preprocessing
    X = data.drop(labels=['price'], axis=1)
    # add feature for categorisation
    X[['year_category', 'mpg_category', 'engine_size_category', 'tax_category']] = None

    # Target + Normalisation
    y = np.log(data['price'])

    transformer = get_transformer(X)

    # testing
    # X_ = pd.DataFrame(transformer.fit_transform(X).toarray())
    # print(X_.info())

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.15)

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
                                  n_estimators=200, verbose=1),
                              verbose=True)
        model.fit(X_train, y_train)
        dump(model, model_path)

    evaluate(model, X_val, y_val)

    # score: 2162.64987495974

    bsp.get_learning_curve(model, X_train, y_train,
                           scoring='neg_mean_squared_error')
