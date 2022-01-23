import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from data_utils import *
from feature_utils import *
from rank_score_utils import *
from itemcf import *
from utils import *

from lgb_utils import train_kFold_lgb_ranking
import joblib
from validate_submission import *
source_market_dict = {
    "t1":"s1",
    "t2":"s1"
}
def solve(data_dir, reload, source_market=''):
    market_name = data_dir.split('/')[-1]
    print('='*40)
    print('solving {}'.format(market_name))
    print('='*40)
    pkl_path = os.path.join(data_dir, 'pkl')
    valid_pkl_path = os.path.join(pkl_path, 'valid.pkl')
    test_pkl_path = os.path.join(pkl_path, 'test.pkl')
    
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    # load train and train5core ready
    d_path = data_dir
    train5core = pd.read_csv(os.path.join(d_path, 'train_5core.tsv'), sep='\t').sort_values('userId')
    train = pd.read_csv(os.path.join(d_path, 'train.tsv'), sep='\t').sort_values('userId')

    # get valid,test lgb format data
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(d_path)
    
    # score
    if reload:
        lgb_valid_data = joblib.load(valid_pkl_path)
        lgb_test_data = joblib.load(test_pkl_path)
    else:
        lgb_valid_data, lgb_test_data = rank_score(lgb_valid_data, lgb_test_data, data_dir)
        if source_market != '':
            source_markets = source_market.split(',')
            for s in source_markets:
                lgb_valid_data, lgb_test_data = rank_score_from_source(lgb_valid_data, lgb_test_data, target=market_name, source=s)
        # joblib.dump(lgb_valid_data, valid_pkl_path)
        # joblib.dump(lgb_test_data, test_pkl_path)

    lgb_valid_data, lgb_test_data = get_train_features(train, train5core, lgb_valid_data, lgb_test_data)
    remove_features_list = ['userId', 'itemId', 'score', 'rating'] 
    features_list = [fea for fea in lgb_valid_data.columns if fea not in remove_features_list]

    final_res = {
        "col":[],
        "ndcg_cut_10":[],
        "recall_10":[]
    }

    for col in features_list:
        print(col)
        ans = offline_scores(lgb_valid_data, col, data_dir, data_dir[-2:])
        final_res['col'].append(col)
        final_res['ndcg_cut_10'].append(ans['ndcg_cut_10'])
        final_res['recall_10'].append(ans['recall_10'])
        
    final_res = pd.DataFrame(final_res)
    final_res = final_res.sort_values('ndcg_cut_10', ascending=False)
    print(final_res)
    save_path = f'../result/imp_{getstrtime()}.csv/'
    # final_res.to_csv(save_path, index=None)
    
    # train 5fold lgb
    assert set(lgb_valid_data.columns) == set(lgb_test_data.columns)
    # print(lgb_valid_data.columns)
    valid_out, test_out = train_kFold_lgb_ranking(lgb_valid_data, lgb_test_data, features_list,Kfold=5)
    return valid_out, test_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1_dir", help="Path to the DATA dir of the kit. Default: ../input/t1", default='../input/t1')
    parser.add_argument("--t2_dir", help="Path to the DATA dir of the kit. Default: ../input/t2", default='../input/t2')
    parser.add_argument("--t1_s", help="source market for t1", default='s2')
    parser.add_argument("--t2_s", help="source market for t2", default='t1')

    parser.add_argument("--reload", help="reload pkl features", action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    valid_out2, test_out2 = solve(args.t2_dir, args.reload, args.t2_s)
    valid_out1, test_out1 = solve(args.t1_dir, args.reload, args.t1_s)

    save_path = f'../result/tmp{getstrtime()}/'
    save(valid_out1, test_out1, valid_out2, test_out2, save_path=save_path)
    print('saved to ', save_path)
if __name__ == '__main__':
    main()

