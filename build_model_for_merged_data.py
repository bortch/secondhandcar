from os.path import join, isfile
import pandas as pd
import numpy as np

from bs_lib.bs_eda import load_all_csv, load_csv_file
from bs_lib.bs_eda import train_val_test_split, get_ordered_categories
import prepare_data as prepare
import build_model as build

if __name__ == "__main__":

    np.random.seed(1)

    verbose = False

    file_to_exclude = []
    current_directory = "."
    dataset_directory = "dataset"
    prefix = 'file_processed'
    file_directory = join(current_directory, dataset_directory, prefix)
    dest_directory = join(current_directory, dataset_directory, prefix)
    filename = "all_set.csv"
    file_path = join(dest_directory, filename)
    
    # ------------------------------------------------------------------------#
    # Load files processed dataset
    # ________________________________________________________________________#
    
    # if file doesn't already exist
    if isfile(file_path):
        all_set_df = load_csv_file(file_path, index=0)
    else:
        all_df_dict = load_all_csv(
            dataset_path=file_directory, exclude=file_to_exclude,index=0)

        columns = all_df_dict['audi'].columns.to_list()
        # load all files
        all_set_df = pd.DataFrame(columns=columns,dtype=float)
        # merge all files
        for brand, dataframe in all_df_dict.items():
            #print(dataframe.info())
            all_set_df=pd.concat([all_set_df,dataframe], ignore_index=True)
        print(all_set_df.info())
        #all_set_df.to_csv(file_path)

    # ------------------------------------------------------------------------#
    # prepare dataset
    # ________________________________________________________________________#
    
    prepared_df = prepare.load_prepared_file(filename=filename)
    if isinstance(prepared_df, pd.DataFrame):
        all_set_df = prepared_df
    else:
    
        all_set_df = prepare.clean_variables(all_set_df)
        all_set_df = prepare.nan_outliers(all_set_df)
        all_set_df = prepare.numerical_imputer(all_set_df, n_neighbors=10,weights='distance', imputer_type='KNN')
        
        prepare.save_prepared_file(all_set_df,filename=filename)
    
    # ------------------------------------------------------------------------#
    # split dataset
    # ________________________________________________________________________#
    
    target = 'price'
    categories = get_ordered_categories(all_set_df, by=target)

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

    
    model = build.get_model()
    params = {
        "transformer__model_pipe__OE__categories": [categories['model']],
        "transformer__brand_pipe__OHE__categories": [categories['brand']],
        "transformer__transmission_pipe__OHE__categories": [categories['transmission']],
        "transformer__fuel_type_pipe__OE__categories": [categories['fuel_type']],
        "transformer__year_pipe":'passthrough'
    }
    # print(model.get_params().keys())
    model.set_params(**params)
    model.fit(X_train,y_train)

    y_pred, y_val, rmse = build.evaluate(model, X_val, y_val)

    build.print_performance([[f"All Brand Dataset",
                   int(round(rmse, 0)),
                   int(round(y_val.mean(), 0)),
                   int(round(y_pred.mean(), 0)),
                   '-']])