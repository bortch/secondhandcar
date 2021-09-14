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
        X[c] = X[c].astype('category')
        X[c] = X[c].cat.codes
        # .astype('category').cat.codes.astype('category')
        # print(f"Categorize: {c}\n\t{type(X[c])}")
        # .astype('category')#.cat.codes
        # .apply(lambda x: x.astype('category').cat.codes)
    print(X.info())
    return X



def get_transformer(X):

    # union = FeatureUnion([("MinMax", MinMaxScaler()),
    #                   ("SS", StandardScaler()),
    #                   ("Log", FunctionTransformer(np.log1p)])
    # proc = ColumnTransformer([('trylots', union, ['Value_In_Dollars'])],
    #                      remainder='passthrough')
    # cut_and_encode = FeatureUnion([("category",)])
    # df['animal_name'].cat.codes.astype('category')

    # cat_columns = X.select_dtypes(include=['object']).columns.tolist()
    # print(cat_columns)
    cat_columns = ['transmission', 'fuel_type']
    categorizer = FunctionTransformer(categorize)
    year_bins = np.arange(2009, 2022)
    mpg_bins = [0, 36, 47, 100]
    engine_bins = [-1, 2, 7]
    tax_bins = [-1, 100, 125, 175, 225, 250, 275, 1000]

    categorical_pipeline = make_pipeline(
        categorizer, OneHotEncoder(handle_unknown='ignore'), verbose=True)

    transformer = make_column_transformer(
        (tsf.TypeConverter('float'), ['mileage']),
        (tsf.Discretizer(target=['year'],
                         kwargs={"bins": year_bins}), ['year']),
        (tsf.Discretizer(target=['mpg'],
         kwargs={"bins": mpg_bins, "labels": ["Low", "Medium", "High"]}), ['mpg']),
        (tsf.Discretizer(target=['engine_size'],
         kwargs={"bins": engine_bins, "labels": ['Small', 'Large']}), ['engine_size']),
        (tsf.Discretizer(target=['tax'],
                         kwargs={"bins": tax_bins}), ['tax']),
        (OrdinalEncoder(), ['year_category', 'mpg_category',
         'engine_size_category', 'tax_category']),
        (categorizer, cat_columns),
        (categorical_pipeline, ['model', 'brand']),
        (StandardScaler(), make_column_selector(dtype_include='float64')),
        remainder='passthrough', verbose=2)
    return transformer

def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_pred)))
    print(rmse)

def fit_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    evaluate(model,X_val,y_val)


if __name__ == "__main__":
    np.random.seed(1)

    dataset_directory_path = 'dataset/'
    model_directory_path = 'model/'
    dataset_file = 'train_set_light_preprocessed.csv'
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
    model_filename = '_temp_model_3.joblib'
    model_path = join(model_directory_path, model_filename)
    if isfile(model_path):
        model = load(model_path)
        print(model)
    else:
        # pipeline: predict preprocessing
        model = make_pipeline(transformer,
                              RandomForestRegressor(n_estimators=200),
                              verbose=True)
        model.fit(X_train, y_train)
        dump(model, model_path)

    #evaluate(model,X_val,y_val)

    # scores = cross_val_score(model, X_train, y_train, cv=5)
    # print(scores.mean())

    bsp.get_learning_curve(model, X_train, y_train, scoring="recall")
