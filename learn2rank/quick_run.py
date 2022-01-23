import argparse
import os
import sys
import numpy as np
import pandas as pd
from data_utils import *
from utils import *
from lgb_utils import train_kFold_lgb_ranking, train_kFold_catboost
import joblib
from validate_submission import *

def main():
    lgb_valid_data = joblib.load('final_valid.pickle')
    lgb_test_data = joblib.load('final_test.pickle')
    
    remove_features_list = ['userId', 'itemId', 'score', 'rating'] 
    features_list = [fea for fea in lgb_valid_data.columns if fea not in remove_features_list]

    valid_out, test_out = train_kFold_lgb_ranking(lgb_valid_data, lgb_test_data, features_list, Kfold=7, cross_domain=True)
    valid_out = valid_out.loc[:,~valid_out.columns.duplicated()]
    test_out = test_out.loc[:,~test_out.columns.duplicated()]
    
    valid_out1, valid_out2 = valid_out[valid_out['m'] == 1], valid_out[valid_out['m'] == 2]
    test_out1,  test_out2  = test_out[test_out['m'] == 1],  test_out[test_out['m'] == 2]
    save_path = f'../result/lgb_ranking/'
    save(valid_out1, test_out1, valid_out2, test_out2, save_path=save_path)
    print('saved to ', save_path)
if __name__ == '__main__':
    main()
