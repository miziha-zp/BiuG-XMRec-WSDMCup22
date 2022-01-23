import os
import zipfile
import time
import joblib

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df, use_float16=True):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def reduce_pair_mem(a, b):
    a = reduce_mem_usage(a)
    b = reduce_mem_usage(b)
    return a, b

def getstrtime():
    output_file='time_'+time.strftime("%m%d%H%M", time.localtime())
    return output_file

def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()
 
import shutil
def save(t1_valid, t1_test, t2_valid, t2_test, save_path='result/temp', remove_oral_file=True):
    '''
     submission.zip
        ├── t1
            ├── test_pred.tsv     /* scores of test items */
            └── valid_pred.tsv    /* scores of validation items */
        ├── t2
            ├── test_pred.tsv     /* scores of test items */
            └── valid_pred.tsv    /* scores of validation items */
    '''
    t1_save_path = os.path.join(save_path, 't1')
    t2_save_path = os.path.join(save_path, 't2')
    if not os.path.exists(t1_save_path):
        os.makedirs(t1_save_path)
    if not os.path.exists(t2_save_path):
        os.makedirs(t2_save_path)

    t1_valid = t1_valid[['userId', 'itemId', 'score']]
    t2_valid = t2_valid[['userId', 'itemId', 'score']]
    t1_test = t1_test[['userId', 'itemId', 'score']]
    t2_test = t2_test[['userId', 'itemId', 'score']]

    t1_valid.to_csv(os.path.join(t1_save_path, 'valid_pred.tsv'), sep='\t', index=None)
    t2_valid.to_csv(os.path.join(t2_save_path, 'valid_pred.tsv'), sep='\t', index=None)
    t1_test.to_csv(os.path.join(t1_save_path, 'test_pred.tsv'), sep='\t', index=None)
    t2_test.to_csv(os.path.join(t2_save_path, 'test_pred.tsv'), sep='\t', index=None)

    
    # zip 
    save_zip_fn = os.path.join(save_path, 'submission.zip')
    # zipDir(save_path, save_zip_fn)

    # remove 
    # if remove_oral_file:
    #     shutil.rmtree(t1_save_path)
    #     shutil.rmtree(t2_save_path)
        # shutil.rmtree(save_path)
        
    
if __name__ == "__main__":
    input_path = "./origin_file_001"
    output_path = "./test.zip"
 
    zipDir(input_path, output_path)