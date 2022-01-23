import pandas as pd
import numpy as np
import os
from math import sqrt
import swifter
from tqdm import tqdm
from os.path import join as pjoin
from collections import defaultdict
from stats_utils import *
def merge_df(df_5core,df_oral,thed=0):
    df = df_oral.copy()
    df['rating'] = df['rating'].apply(lambda x:1 if x>= thed else 0)
    df = df.loc[df['rating']==1]
    data = pd.concat([df, df_5core])
    data.drop_duplicates(subset=['userId','itemId'],keep='first',inplace=True)
    return data

def df2dict(data):
    i2us, u2is = defaultdict(set), defaultdict(set)
    for u, i in zip(data['userId'].values, data['itemId'].values):
        i2us[i].add(u)
        u2is[u].add(i)

    return i2us, u2is

def load_graph(data_path, merge_train=True):
    train5core = pd.read_csv(os.path.join(data_path, 'train_5core.tsv'), sep='\t').sort_values('userId')
    train = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t').sort_values('userId')
    if merge_train:
        data = merge_df(train5core, train)
    else:
        data = train5core

    return data

def cal_IOU(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) /  (len(union)+1e-7)

def cosine_similarity(set1, set2):
    intersection = set1 & set2
    return len(intersection) / (sqrt(len(set1) * len(set2))+1e-7)

def get_stats(data, col_name, prefix):
    data[col_name] = data[col_name].apply(np.array)
    
    # numpy based function
    stat_funtion = [np.max, np.mean, np.sum, np.std, \
        root_mean_square, len,]
  
    stat_name = ['max', 'mean','sum', 'std', \
        "root_mean_square", 'length']

    for stat_n, stat_f in zip(stat_name, stat_funtion):
        print('processing ====> ', prefix, stat_n)
        data[prefix+'_'+stat_n] = data[col_name].apply(stat_f)
    # pandas based function
    # TODO BUG
    '''
        File "/home/data/zp/gll/XMRec-WSDM22/2022/stats_utils.py", line 7, in q10
        return x.quantile(0.1)
        AttributeError: 'float' object has no attribute 'quantile'
    '''
    # data[col_name] = data[col_name].apply(pd.Series)
    stat_funtion = [q10, q20, q25, q30, q40, q60, q70, q75, q80, q90,]   
    # stat_name = ['q10', 'q20', 'q25', 'q30', 'q40', 'q60', 'q70', 'q75', 'q80', 'q90'] 
    # for stat_n, stat_f in zip(stat_name, stat_funtion):
    #     print('processing ====> ', prefix, stat_n)
    #     data[prefix+'_'+stat_n] = data[col_name].apply(stat_f)

    del data[col_name]
    return data

def cal_i2u_u(data, u2is, i2us, prefix='i2u_u', sim='iou'):
    '''
    Paramters:
    sim: 'iou' or 'cosine'
    '''
    prefix += sim 
    assert sim in ['iou', 'cosine']
    sim_fun = cal_IOU if sim == 'iou' else cosine_similarity
    i2i_score_list = []
    for u, i in tqdm(zip(data['userId'].values, data['itemId'].values)):
        users = i2us[i]
        ans_list = [sim_fun(u2is[tu], u2is[u])for tu in users]
        # handle cold start item
        if len(ans_list) == 0:
            ans_list = [-1, -1]
        i2i_score_list.append(ans_list)
    col_name = prefix + '_list'
    data[col_name] = i2i_score_list
    print('='*40)
    print('stats for {}'.format(prefix))
    print('='*40) 
    data = get_stats(data, col_name, prefix)
    return data

def SwingRecall(u2items):
    u2Swing = defaultdict(lambda:dict())
    for u in tqdm(u2items):
        wu = pow(len(u2items[u])+5,-0.35)
        for v in u2items:
            if v == u:
                continue
            wv = wu*pow(len(u2items[v])+5,-0.35)
            inter_items = set(u2items[u]).intersection(set(u2items[v]))
            for i in inter_items:
                for j in inter_items:
                    if j==i:
                        continue
                    if j not in u2Swing[i]:
                        u2Swing[i][j] = 0
                    u2Swing[i][j] += wv/(1 + len(inter_items))
    return u2Swing

def cal_swing(data, u2is, i2us, sim, prefix='swing', alpha=0.5):
    '''
    Paramters:
    sim: 'iou' or 'cosine'
    '''
    # def cal_sim_swing(i1, i2):
    #     ans = 0
    #     i1_and_i2 = i2us[i1] & i2us[i2]
    #     for u_i in i1_and_i2:
    #         for u_j in i1_and_i2:
    #             if u_i != u_j:
    #                 ans += 1 / (len(u2is[u_i]&u2is[u_j])+alpha)
    #     return ans
    def cal_sim_swing(i1, i2, sim):
        if i1 not in sim or i2 not in sim[i1]:
            return 0
        return sim[i1][i2]

    
    i2i_score_list = []
    for u, i in tqdm(zip(data['userId'].values, data['itemId'].values)):
        items = u2is[u]
        ans_list = [cal_sim_swing(ti, i, sim)for ti in items]
        # handle cold start user
        if len(ans_list) == 0:
            ans_list = [-1, -1]
        i2i_score_list.append(ans_list)
    col_name = prefix + '_list'
    data[col_name] = i2i_score_list
    print('='*40)
    print('stats for {}'.format(prefix))
    print('='*40) 
    data = get_stats(data, col_name, prefix)
    return data

def cal_u2i_i(data, u2is, i2us, prefix='u2i_i', sim='iou'):
    '''
    Paramters:
    sim: 'iou' or 'cosine'
    '''
    prefix += sim 
    assert sim in ['iou', 'cosine']
    sim_fun = cal_IOU if sim == 'iou' else cosine_similarity
    i2i_score_list = []
    for u, i in tqdm(zip(data['userId'].values, data['itemId'].values)):
        items = u2is[u]
        ans_list = [sim_fun(i2us[ti], i2us[i])for ti in items]
        # handle cold start user
        if len(ans_list) == 0:
            ans_list = [-1, -1]
        i2i_score_list.append(ans_list)
    col_name = prefix + '_list'
    data[col_name] = i2i_score_list
    print('='*40)
    print('stats for {}'.format(prefix))
    print('='*40)
    data = get_stats(data, col_name, prefix)
    return data

def rank_score_from_source(lgb_valid_data, lgb_test_data, target='t1', source='s1', data_dir='../input/'):
    
    sdata_path = pjoin(data_dir, source)
    sdata = load_graph(sdata_path)
    i2us, _ = df2dict(sdata)
    # watch out! u2is must get from target market
    ddata_path = pjoin(data_dir, target)
    ddata = load_graph(ddata_path)
    _, u2is = df2dict(ddata)

    # lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='{}_u2i'.format(source), sim='iou')
    # lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='{}_u2i'.format(source), sim='iou')
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='{}_u2i'.format(source), sim='cosine')
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='{}_u2i'.format(source), sim='cosine')

    return lgb_valid_data, lgb_test_data

def rank_score(lgb_valid_data, lgb_test_data, data_dir):
    '''
    '''
    market_name = data_dir.split('/')[-1]
    print('='*40)
    print('rank score for {}'.format(market_name))
    print('='*40)
    data = load_graph(data_dir)
    i2us, u2is = df2dict(data)
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='u2i', sim='iou')
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='u2i', sim='iou')
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='u2i', sim='cosine')
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='u2i', sim='cosine')
    sim = SwingRecall(u2is)
    lgb_valid_data = cal_swing(lgb_valid_data, u2is, i2us, sim, prefix='swing')
    lgb_test_data = cal_swing(lgb_test_data, u2is, i2us, sim, prefix='swing')

    lgb_valid_data = cal_i2u_u(lgb_valid_data, u2is, i2us, prefix='i2u', sim='iou')
    lgb_test_data = cal_i2u_u(lgb_test_data, u2is, i2us, prefix='i2u', sim='iou')
    lgb_valid_data = cal_i2u_u(lgb_valid_data, u2is, i2us, prefix='i2u', sim='cosine')
    lgb_test_data = cal_i2u_u(lgb_test_data, u2is, i2us, prefix='i2u', sim='cosine')

    return lgb_valid_data, lgb_test_data