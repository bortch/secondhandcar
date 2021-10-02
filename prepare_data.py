from os.path import join, isfile
import pandas as pd
import numpy as np
from bs_lib.bs_eda import load_all_csv
from bs_lib.bs_eda import get_numerical_columns, get_categorical_columns, train_val_test_split, split_by_row, load_csv_files_as_dict

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from scipy.stats import zscore
import constants as cnst

def set_as_categorical(data):
    df = data.copy()
    print("\t\tChange type to Categorical")
    columns = get_categorical_columns(df)
    for c in columns:
        df[c] = pd.Categorical(df[c], categories=df[c].unique().tolist())
    print(f"columns set as category {columns}")
    return df


def set_as_numerical(data):
    df = data.copy()
    num_columns = get_numerical_columns(df)
    # df['price']=df['price']/1.
    for c in num_columns:
        df[c] = pd.to_numeric(df[c])
    return df


def clean_variables(data):
    print("\n\tRemoving duplicate entries and noisy features:")
    df = data.copy()
    df = set_as_categorical(df)

    # remove duplicate
    print("\t\tDrop duplicate")
    df.drop_duplicates(inplace=True)

    df = set_as_numerical(df)

    # scale
    print("\t\tScaling numerical feature")
    num_columns = get_numerical_columns(df)
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
    # log target
    print("Replace target by Log(target)")
    df['price'] = np.log(df['price'])
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
        # print(df.info())
    return df


def numerical_imputer(data, n_neighbors=10, weights='distance', fit_set=None, imputer_type=None):
    print('\nImput numerical missing value')
    df = data.copy()
    # print("df",df.info())
    columns = get_numerical_columns(df)
    if 'price' in columns:
        columns.remove('price')
    has_nan = df.isnull().values.any()
    print(f"\t{columns} has NAN? {has_nan}")
    if(has_nan):
        print("\tNAN found, imputing ...")
        if imputer_type == 'KNN':
            print('\tusing KNNImputer')
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        else:
            print('\tUsing IterativeImputer')
            imputer = IterativeImputer(random_state=0)
        if isinstance(fit_set, pd.DataFrame):
            print('\tfit imputer using fit_set:', fit_set.shape)
            imputer.fit(fit_set[columns])
            imputed = imputer.transform(df[columns])
        else:
            imputed = imputer.fit_transform(df[columns])
        for i, c in enumerate(columns):
            df[c] = imputed[:, i]
        #print("\thas inf?", df.isin([np.nan, np.inf, -np.inf]).values.any())
    print("\tImputation done?", not df.isnull().values.any())
    return df


def save_prepared_file(df, filename):
    dest_file_path = join(cnst.PREPARED_DATA_PATH, filename)
    df.to_csv(dest_file_path)
    print(f"{filename} data saved @ {dest_file_path}")


def load_prepared_file(filename):
    file_path = join(cnst.PREPARED_DATA_PATH, filename)
    if isfile(file_path):
        #print(f"Loading {file_path}")
        df = pd.read_csv(file_path, index_col=0)
        #df = set_as_categorical(df)
        return df
    else:
        return None


if __name__ == "__main__":
    file_to_exclude = ['all_set.csv']

    all_df = load_all_csv(dataset_path=cnst.FILE_PROCESSED_PATH, exclude=file_to_exclude)

    columns = ['model', 'year',
               'price', 'transmission',
               'mileage', 'fuel_type',
               'tax', 'mpg',
               'engine_size', 'brand']

    for brand, dataframe in all_df.items():
        print(f"\nPreparing {brand} data\nShape: {dataframe.shape}")
        _df = pd.DataFrame(all_df[brand], columns=columns)

        # Cleaning step
        _df = clean_variables(_df)

        # Outlier
        #fit_set = drop_outliers(df_)
        _df = nan_outliers(_df)
        _df = numerical_imputer(_df, n_neighbors=10,
                                weights='distance', imputer_type='KNN')

        save_prepared_file(_df, filename=f"{brand}.csv")
