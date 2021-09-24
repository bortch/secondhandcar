from os.path import join
import pandas as pd
import numpy as np
from bs_lib.bs_eda import load_all_csv
from bs_lib.bs_eda import get_numerical_columns, get_categorical_columns, train_val_test_split, split_by_row, load_csv_files_as_dict


if __name__ == "__main__":
    file_to_exclude = []
    current_directory = "."
    dataset_directory = "dataset"
    file_directory = join(current_directory,dataset_directory,'prepared_data')
    dest_directory = join(current_directory,dataset_directory,'splitted_data')

    all_df = load_all_csv(dataset_path=file_directory, exclude=file_to_exclude)
    
    columns = ['model', 'year',
               'price', 'transmission',
               'mileage', 'fuel_type',
               'tax', 'mpg',
               'engine_size', 'brand']

    for brand, dataframe in all_df.items():
        print(f"\nPreparing {brand} data\nShape: {dataframe.shape}")
        _df = pd.DataFrame(all_df[brand],columns=columns)
        
        # Cleaning step
        X_train, X_val, X_test, y_train, y_val, y_test= train_val_test_split(X=_df[brand],
                                                                              y=_df[brand][target], 
                                                                              train_size=.75, 
                                                                              val_size=.15, 
                                                                              test_size=.1, 
                                                                              random_state=1,
                                                                              show=True)
        dest_file_path = join(dest_directory,f'{brand}.csv')
        _df.to_csv(dest_file_path)
        print(f"{brand} data saved @ {dest_file_path}")