import os
import joblib
from os.path import join as pjoin
import pandas as pd
import numpy as np
from utils import *
from validate_submission import offline_scores
def save(ids_valid_data, ids_test_data, data_path, description='test',testing=False):
    testing_des = 'testing_' if testing else ''
    pkl_path = pjoin(data_path, f'{testing_des}{description}_score.pkl')
    print('saving ====> {}'.format(pkl_path))
    joblib.dump([ids_valid_data, ids_test_data], pkl_path)


def EASE_r_predict(model, data_path, testing=False, desp=''):
    market_name = os.path.split(data_path)[-1]
    print('===> market: {}'.format(market_name))
    print('===> predict:')
    id_seriral_pkl_path = pjoin(data_path, 'id_seriral.pkl')
    pkl_arr = joblib.load(id_seriral_pkl_path)
    ids_valid_data, ids_test_data = pkl_arr
    # predict 
    final_name = f'EASE_r{desp}'
    def predict_dot_product_score(df, model):
        # TODO: device
        from copy import deepcopy
        users = deepcopy(df['userId'])
        items = deepcopy(df['itemId'])
        scores = []
        
        scores = model.predict(users, items)
        df[f'{final_name}_scores'] = scores
        return df

    ids_valid_data = predict_dot_product_score(ids_valid_data, model)
    ids_test_data = predict_dot_product_score(ids_test_data, model)
    
    # reverse_ids
    def return_idx(df, itemIdsmap, userIdsmap):
        df['userId'] = userIdsmap.id2idx(df['userId'])
        df['itemId'] = itemIdsmap.id2idx(df['itemId'])
        return df

    print('====> load IdsMap')
    print(pjoin(data_path, "itemIdsMap.pkl"))
    itemIdsMap = joblib.load(pjoin(data_path, "itemIdsMap.pkl"))
    userIdsMap = joblib.load(pjoin(data_path, "userIdsMap.pkl"))
    ids_valid_data = return_idx(ids_valid_data, itemIdsMap, userIdsMap)
    ids_test_data = return_idx(ids_test_data, itemIdsMap, userIdsMap)
    # cal score from groudtruth.
    offline_scores(ids_valid_data, f'{final_name}_scores',data_path, market_name)
    # save score as features for lightgbm
    save(ids_valid_data, ids_test_data, data_path, description=f'{final_name}_scores',testing=testing)


def SLIM_predict(model, data_path, testing=False):
    market_name = os.path.split(data_path)[-1]
    print('===> market: {}'.format(market_name))
    print('===> predict:')
    id_seriral_pkl_path = pjoin(data_path, 'id_seriral.pkl')
    pkl_arr = joblib.load(id_seriral_pkl_path)
    ids_valid_data, ids_test_data = pkl_arr
    # predict 
    def predict_dot_product_score(df, model):
        # TODO: device
        users = df['userId']
        items = df['itemId']
        scores = []
        for u, i in zip(users, items):
            try:
                scores.append(model.predict(u, i))
            except:
                scores.append(0)
        df['SLIM_scores'] = scores
        return df

    ids_valid_data = predict_dot_product_score(ids_valid_data, model)
    ids_test_data = predict_dot_product_score(ids_test_data, model)
    
    # reverse_ids
    def return_idx(df, itemIdsmap, userIdsmap):
        df['userId'] = userIdsmap.id2idx(df['userId'])
        df['itemId'] = itemIdsmap.id2idx(df['itemId'])
        return df

    print('====> load IdsMap')
    print(pjoin(data_path, "itemIdsMap.pkl"))
    itemIdsMap = joblib.load(pjoin(data_path, "itemIdsMap.pkl"))
    userIdsMap = joblib.load(pjoin(data_path, "userIdsMap.pkl"))
    ids_valid_data = return_idx(ids_valid_data, itemIdsMap, userIdsMap)
    ids_test_data = return_idx(ids_test_data, itemIdsMap, userIdsMap)
    # cal score from groudtruth.
    offline_scores(ids_valid_data, 'SLIM_scores',data_path, market_name)
    # save score as features for lightgbm
    save(ids_valid_data, ids_test_data, data_path, description='SLIM_scores',testing=testing)


def item2vec_predict(model, data_path, dim=40, testing=False):
    market_name = os.path.split(data_path)[-1]
    print('===> market: {}'.format(market_name))
    print('===> predict:')
    id_seriral_pkl_path = pjoin(data_path, 'id_seriral.pkl')
    pkl_arr = joblib.load(id_seriral_pkl_path)
    ids_valid_data, ids_test_data = pkl_arr
    # predict 
    def predict_dot_product_score(df, model, dim):
        # TODO: device
        users = df['userId']
        items = df['itemId']
        scores = []
        for u, i in zip(users, items):
            try:
                scores.append(model.predict(u, i))
            except:
                scores.append(0)
        df[f'item2vec{dim}_scores'] = scores
        return df

    ids_valid_data = predict_dot_product_score(ids_valid_data, model, dim)
    ids_test_data = predict_dot_product_score(ids_test_data, model, dim)
    
    # reverse_ids
    def return_idx(df, itemIdsmap, userIdsmap):
        df['userId'] = userIdsmap.id2idx(df['userId'])
        df['itemId'] = itemIdsmap.id2idx(df['itemId'])
        return df

    print('====> load IdsMap')
    print(pjoin(data_path, "itemIdsMap.pkl"))
    itemIdsMap = joblib.load(pjoin(data_path, "itemIdsMap.pkl"))
    userIdsMap = joblib.load(pjoin(data_path, "userIdsMap.pkl"))
    ids_valid_data = return_idx(ids_valid_data, itemIdsMap, userIdsMap)
    ids_test_data = return_idx(ids_test_data, itemIdsMap, userIdsMap)
    # cal score from groudtruth.
    offline_scores(ids_valid_data, f'item2vec{dim}_scores',data_path, market_name)
    # save score as features for lightgbm
    # features_ultraGCN_pkl_path = pjoin(data_path, 'item2vec_scores{}.pkl'.format(K))
    # print('saving ====> '.format(features_ultraGCN_pkl_path))
    # joblib.dump([ids_valid_data, ids_test_data], features_ultraGCN_pkl_path)
    save(ids_valid_data, ids_test_data, data_path, description=f'item2vec{dim}',testing=testing)


def itemKNN_predict(model, data_path, K=40, testing=False):
    market_name = os.path.split(data_path)[-1]
    print('===> market: {}'.format(market_name))
    print('===> predict:')
    id_seriral_pkl_path = pjoin(data_path, 'id_seriral.pkl')
    pkl_arr = joblib.load(id_seriral_pkl_path)
    ids_valid_data, ids_test_data = pkl_arr
    # predict 
    def predict_dot_product_score(df, model, K):
        # TODO: device
        users = df['userId']
        items = df['itemId']
        scores = []
        for u, i in zip(users, items):
            try:
                scores.append(model.score(u, i))
            except:
                scores.append(0)
        df[f'itemKNN{K}_scores'] = scores
        return df

    ids_valid_data = predict_dot_product_score(ids_valid_data, model, K)
    ids_test_data = predict_dot_product_score(ids_test_data, model, K)
    
    # reverse_ids
    def return_idx(df, itemIdsmap, userIdsmap):
        df['userId'] = userIdsmap.id2idx(df['userId'])
        df['itemId'] = itemIdsmap.id2idx(df['itemId'])
        return df

    print('====> load IdsMap')
    print(pjoin(data_path, "itemIdsMap.pkl"))
    itemIdsMap = joblib.load(pjoin(data_path, "itemIdsMap.pkl"))
    userIdsMap = joblib.load(pjoin(data_path, "userIdsMap.pkl"))
    ids_valid_data = return_idx(ids_valid_data, itemIdsMap, userIdsMap)
    ids_test_data = return_idx(ids_test_data, itemIdsMap, userIdsMap)
    # cal score from groudtruth.
    offline_scores(ids_valid_data, f'itemKNN{K}_scores',data_path, market_name)
    save(ids_valid_data, ids_test_data, data_path, description=f'itemKNN{K}_scores',testing=testing)

