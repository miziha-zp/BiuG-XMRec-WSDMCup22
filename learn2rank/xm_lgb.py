import argparse
import os
import sys
import numpy as np
import pandas as pd

from data_utils import *
from feature_utils import *
from itemcf import *
from utils import *
from swing import *
from itemcf_cross import get_crossitemCF_score
from lgb_utils import train_kFold_lgb_ranking, train_kFold_catboost
import joblib
from validate_submission import *
from xm_cf.rank_score_utils import *
source_market = {
    "t1":"s1,s2,s3,all",
    "t2":"s1,s2,s3,all"
    # "t1":"all",
    # "t2":"all"
}

source_marketsss = {
    # 't1': ['t1-t2-s1-s2-s3', 's1-t1', 's2-t1', 's3-t1'],
    # 't2': ['t1-t2-s1-s2-s3', 's1-t2', 's2-t2', 's3-t2'],
    't1':['t1-t2-s1-s2-s3'],
    't2':['t1-t2-s1-s2-s3']
}
np.random.seed(2022)
def get_offline_score(lgb_valid_data, data_dir, market_name, save_path='tmp'):
    # print(lgb_valid_data.head())
    remove_features_list = ['userId', 'itemId', 'score', 'rating'] 
    features_list = [fea for fea in lgb_valid_data.columns if fea not in remove_features_list]
   
    # train 5fold lgb
    # assert set(lgb_valid_data.columns) == set(lgb_test_data.columns)
    # print(lgb_valid_data.columns)

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
    save_path = f'{save_path}imp_lgb_{market_name}.csv'
    final_res.to_csv(save_path, index=None)

def solve(data_dir, reload, offline):
    market_name = data_dir.split('/')[-1]
    pkl_path = os.path.join(data_dir, 'pkl')
    valid_pkl_path = os.path.join(pkl_path, 'valid.pkl')
    test_pkl_path = os.path.join(pkl_path, 'test.pkl')
    
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    # load train and train5core ready
    d_path = data_dir
    train5core = pd.read_csv(os.path.join(d_path, 'train_5core.tsv'), sep='\t').sort_values('userId')
    train = pd.read_csv(os.path.join(d_path, 'train.tsv'), sep='\t').sort_values('userId')
    # get valid, test lgb format data
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(d_path)
    # w2v features from train5core and train
    lgb_valid_data, lgb_test_data = get_w2v_features(train, train5core, lgb_valid_data, lgb_test_data)
    # print(lgb_valid_data.columns)
    # get_offline_score(lgb_valid_data, data_dir, market_name)
    if not reload:
        # itemcf,swingcf,usercf
        lgb_valid_data, lgb_test_data = rank_score(lgb_valid_data, lgb_test_data, data_dir)
        for mstr in source_marketsss[market_name]:
            ms = mstr.split('-')
            for latent_dim in [16, 6, 8]:
                lgb_valid_data, lgb_test_data = get_item2vec_score(lgb_valid_data, lgb_test_data, ms, market_name, latent_dim)

        # get_offline_score(lgb_valid_data, data_dir, market_name)

        if source_market[market_name] != '':
            source_markets = source_market[market_name].split(',')
            for s in source_markets:
                lgb_valid_data, lgb_test_data = rank_score_from_source(lgb_valid_data, lgb_test_data, target=market_name, source=s, concat=True)
                lgb_valid_data, lgb_test_data = rank_score_from_source(lgb_valid_data, lgb_test_data, target=market_name, source=s, concat=False)
                lgb_valid_data, lgb_test_data = rank_score_from_source2(lgb_valid_data, lgb_test_data, target=market_name, source=s, concat=True)
                lgb_valid_data, lgb_test_data = rank_score_from_source2(lgb_valid_data, lgb_test_data, target=market_name, source=s, concat=False)
                
                lgb_valid_data, lgb_test_data = reduce_pair_mem(lgb_valid_data, lgb_test_data)

        joblib.dump(lgb_valid_data, valid_pkl_path)
        joblib.dump(lgb_test_data, test_pkl_path)
    else:
        a = 2
        # lgb_valid_data = joblib.load(valid_pkl_path)
        # lgb_test_data = joblib.load(test_pkl_path)
    # del lgb_test_data['u2iiou_length'], lgb_valid_data['u2iiou_length']
    # print(lgb_valid_data['i2ucosine_max'].head(500))
    # merge train train5core stats features
    # lgb_valid_data, lgb_test_data = get_train_features(train, train5core, lgb_valid_data, lgb_test_data)
    # lgcn_score 655->648
    lgb_valid_data, lgb_test_data = load_lightgcn(data_dir, lgb_valid_data, lgb_test_data)
    # mf 0.6707 -> 6699(remove)
    lgb_valid_data, lgb_test_data = load_mf(data_dir, lgb_valid_data, lgb_test_data)
    # load_slim
    # get_offline_score(lgb_valid_data, data_dir, market_name, '')
    lgb_valid_data, lgb_test_data = load_itemEASE_r(data_dir, lgb_valid_data, lgb_test_data)

    lgb_valid_data, lgb_test_data = load_slim(data_dir, lgb_valid_data, lgb_test_data)
    # ultragcn
    lgb_valid_data, lgb_test_data = load_ultragcn(data_dir, lgb_valid_data, lgb_test_data)
    
    # global features 655->6556
    lgb_valid_data, lgb_test_data = load_global_features(data_dir, lgb_valid_data, lgb_test_data)
    # w2v
    lgb_valid_data, lgb_test_data = reduce_pair_mem(lgb_valid_data, lgb_test_data)
    # itemKNN
    lgb_valid_data, lgb_test_data = load_itemKNN(data_dir, lgb_valid_data, lgb_test_data)
    lgb_valid_data, lgb_test_data = reduce_pair_mem(lgb_valid_data, lgb_test_data)
    # lgbide
    lgb_valid_data, lgb_test_data = load_lgbide(data_dir, lgb_valid_data, lgb_test_data)
    # gfcf
    lgb_valid_data, lgb_test_data = load_gfcf(data_dir, lgb_valid_data, lgb_test_data)
    # lgb_valid_data['sum_mf_lgcn'] = 0.7 * lgb_valid_data['lgcn_score_score'] + 0.3 * lgb_valid_data['mf_score_score']
    # lgb_test_data['sum_mf_lgcn'] = 0.7 * lgb_test_data['lgcn_score_score'] + 0.3 * lgb_test_data['mf_score_score']
    
    # for fea in ["u2iiou_max", "itemKNN2_scores", "concat_s1_u2iiou_max"]:
    #     lgb_valid_data[f'sum_lgcn_{fea}'] = 0.7 * lgb_valid_data['lgcn_score_score']+ 0.3 * lgb_valid_data[fea]
    #     lgb_test_data[f'sum_lgcn_{fea}'] = 0.7 * lgb_test_data['lgcn_score_score'] + 0.3 * lgb_test_data[fea]
    
    # norm features
    # lgb_valid_data, lgb_test_data = norm_feature(lgb_valid_data, lgb_test_data)
    # features from valid_table 6556->0.6486
    # lgb_valid_data, lgb_test_data = get_feature_here(lgb_valid_data, lgb_test_data)
    if offline:
        get_offline_score(lgb_valid_data, data_dir, market_name, '')

    remove_features_list = ['userId', 'itemId', 'score', 'rating']
    print(remove_features_list)
    features_list = [fea for fea in lgb_valid_data.columns if fea not in remove_features_list]
    print(lgb_test_data.columns)    
    return lgb_valid_data.dropna(), lgb_test_data.dropna()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1_dir", help="Path to the DATA dir of the kit. Default: ../input/t1", default='../input/t1')
    parser.add_argument("--t2_dir", help="Path to the DATA dir of the kit. Default: ../input/t2", default='../input/t2')
    parser.add_argument("--reload", help="reload pkl features", action='store_true')
    parser.add_argument("--valid", help="print offline score", action='store_true')
    parser.add_argument("--offline", help="offlinescore for every features", action='store_true')
    
    args = parser.parse_args()
    print(args)
    valid_out1, test_out1 = solve(args.t1_dir, args.reload, args.offline)
    valid_out2, test_out2 = solve(args.t2_dir, args.reload, args.offline)

    valid_out2['m'], test_out2['m'] = 2, 2
    valid_out1['m'], test_out1['m'] = 1, 1

    print("valid_out1:", valid_out1.shape)
    print("valid_out2:", valid_out2.shape)
    
    lgb_valid_data = pd.concat([valid_out1, valid_out2])

    print(lgb_valid_data.shape)
    lgb_test_data = pd.concat([test_out1, test_out2])
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
