from os.path import join

CURRENT_DIRNAME = '.'

DATASET_DIRNAME = 'dataset'
ORIGINAL_DATA_DIRNAME = 'original'
PREPARED_DATA_DIRNAME = 'prepared_data'
FILE_PROCESSED_DIRNAME = 'file_processed'

DATASET_DIR_PATH = join(CURRENT_DIRNAME,DATASET_DIRNAME)
PREPARED_DATA_PATH = join(DATASET_DIR_PATH,PREPARED_DATA_DIRNAME)
FILE_PROCESSED_PATH = join(DATASET_DIR_PATH,FILE_PROCESSED_DIRNAME)
ORIGINAL_DATA_PATH = join(DATASET_DIR_PATH,ORIGINAL_DATA_DIRNAME)

MODEL_DIRNAME = 'model'
MODEL_DIR_PATH = join(CURRENT_DIRNAME,MODEL_DIRNAME)