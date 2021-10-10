from os.path import join
import pandas as pd
import numpy as np
from bs_lib.bs_eda import load_all_csv
import constants as cnst

if __name__ == "__main__":
    file_to_exclude = ['cclass.csv','unclean focus.csv','unclean cclass.csv','focus.csv']
    
    all_df = load_all_csv(dataset_path=cnst.ORIGINAL_DATA_PATH, exclude=file_to_exclude)
    
    columns = ['model', 'year',
               'price', 'transmission',
               'mileage', 'fuel_type',
               'tax', 'mpg',
               'engine_size', 'brand']

    # brand typo correction
    all_df['hyundai'] = all_df.pop('hyundi')
    all_df['mercedes'] = all_df.pop('merc')
    all_df['opel'] = all_df.pop('vauxhall')

    # merge column 'tax' and 'tax(£)' for Hyundai
    all_df['hyundai']['tax'] = all_df['hyundai']['tax(£)']
    all_df['hyundai'].drop(labels='tax(£)', axis=1, inplace=True)

    for brand, dataframe in all_df.items():
        print(f"Source: {brand}, Shape: {dataframe.shape}")
        # Add brand name as feature
        all_df[brand]['brand'] = brand
        # sanitize `model` blank space
        all_df[brand]['model'] = all_df[brand]['model'].str.strip()
        brand_df = pd.DataFrame(all_df[brand],columns=columns)
        dest_file_path = join(cnst.FILE_PROCESSED_PATH,f'{brand}.csv')
        brand_df.to_csv(dest_file_path)
        print(f"{brand} data saved @ {dest_file_path}")