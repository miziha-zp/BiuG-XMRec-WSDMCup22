import pandas as pd
import numpy as np
import os
import math
from math import sqrt
import swifter
from tqdm import tqdm
from os.path import join as pjoin
from collections import defaultdict
from stats_utils import *
from gensim.models import Word2Vec
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

def load_graph(data_path, merge_train=True, merge_valid=True):
    train5core = pd.read_csv(os.path.join(data_path, 'train_5core.tsv'), sep='\t').sort_values('userId')
    train = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t').sort_values('userId')
    valid = pd.read_csv(os.path.join(data_path, 'valid_qrel.tsv'), sep='\t').sort_values('userId')
    data = train5core

    if merge_train:
        data = merge_df(train5core, train)
    if merge_valid:
        data = merge_df(data, valid)
    
    return data


def load_all_graph(data_path, merge_train=True, merge_valid=True, exlude=''):
    data_all = pd.DataFrame()
    print('load all graph')
    for market in ['s1', 's2', 's3', 't1', 't2']:
        if market == exlude:continue
        sdata_path = pjoin(data_path, market)
        data = load_graph(sdata_path, merge_train, merge_valid)
        data_all = pd.concat([data, data_all], axis=0)
    return data_all

def load_multi_graph(data_path, markets, merge_train=True, merge_valid=True):
    data_all = pd.DataFrame()
    print('load multi graph')
    for market in markets:
        testing = merge_valid
        print(f'load from {market}')
        if 's' in market:testing=True
        sdata_path = pjoin(data_path, market)
        data = load_graph(sdata_path, merge_train, merge_valid=testing)
        data_all = pd.concat([data, data_all], axis=0)
    return data_all

def cal_IOU(set1, set2, degree=0):
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) /  (len(union)+1e-7)

def cosine_similarity(set1, set2, degree=0):
    if degree == 0:
        intersection = set1 & set2
        return len(intersection) / (sqrt(len(set1) * len(set2))+1e-7)
    else:
        ans = 0
        for i in set1 & set2:
            ans += degree[i]
        return ans / (sqrt(len(set1) * len(set2))+1e-7)

def get_stats(data, col_name, prefix):
    data[col_name] = data[col_name].apply(np.array)
    
    # numpy based function

    stat_funtion = [np.max, np.mean, len]
    stat_name = ['max', 'mean', 'length']

    for stat_n, stat_f in zip(stat_name, stat_funtion):
        print('processing ====> ', prefix, stat_n)
        data[prefix+'_'+stat_n] = data[col_name].apply(stat_f)
    # data[prefix+'_'+"percentile_90"] = data[col_name].apply(lambda x: np.percentile(x, 90))
    data[prefix+'_'+"percentile_95"] = data[col_name].apply(lambda x: np.percentile(x, 95))
    # data[prefix+'_'+"percentile_80"] = data[col_name].apply(lambda x: np.percentile(x, 80))
    data[prefix+'_'+"percentile_5"] = data[col_name].apply(lambda x: np.percentile(x, 5))
    # data[prefix+'_'+"percentile_25"] = data[col_name].apply(lambda x: np.percentile(x, 25))
    # data[prefix+'_'+"percentile_55"] = data[col_name].apply(lambda x: np.percentile(x, 55))
    data[prefix+'_'+"median"] = data[col_name].apply(lambda x: np.median(x))

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
    itemdegree = {ttt: len(i2us[ttt])for ttt in i2us}
    userdegree = {ttt: len(u2is[ttt])for ttt in u2is}
    for u, i in tqdm(zip(data['userId'].values, data['itemId'].values)):
        users = i2us[i]
        # if u in users:
        #     ans_list = [sim_fun(u2is[tu], u2is[u]-{i})for tu in users if tu != u]
        # else:
        ans_list = [sim_fun(u2is[tu], u2is[u]) for tu in users]
        # handle cold start item
        if len(ans_list) == 0:
            ans_list = [-2]
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
    itemdegree = {ttt: len(i2us[ttt])for ttt in i2us}
    userdegree = {ttt: len(u2is[ttt])for ttt in u2is}
    for u, i in tqdm(zip(data['userId'].values, data['itemId'].values)):
        items = u2is[u]
        ans_list = [cal_sim_swing(ti, i, sim)for ti in items if ti != i]
        # handle cold start user
        if len(ans_list) == 0:
            ans_list = [-2]
        i2i_score_list.append(ans_list)
    col_name = prefix + '_list'
    data[col_name] = i2i_score_list
    print('='*40)
    print('stats for {}'.format(prefix))
    print('='*40) 
    data = get_stats(data, col_name, prefix)
    return data

def get_i2i_sim(user_item_time_dict):
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc_1, (i, i_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            sim_item.setdefault(i, {})
            for loc_2, (relate_item, related_time) in enumerate(item_time_list):
                if i == relate_item:
                    continue
                loc_alpha = 1.0 if loc_2 > loc_1 else 0.7
                loc_weight = loc_alpha * (0.8 ** (np.abs(loc_2 - loc_1) - 1))
                time_weight = np.exp(-15000 * np.abs(i_time - related_time))

                sim_item[i].setdefault(relate_item, 0)
                sim_item[i][relate_item] += loc_weight * time_weight / math.log(1 + len(item_time_list))

    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])
    return sim_item

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
        # ans_list = [sim_fun(i2us[ti]-{u}, i2us[i]-{u})for ti in items if ti != i]
        ans_list = [sim_fun(i2us[ti], i2us[i])for ti in items]
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

def rank_score_from_source2(lgb_valid_data, lgb_test_data, target='t1', source='s1', data_dir='../input/', concat=True):
    concat_desc = 'concat' if concat else ''
    sdata_path = pjoin(data_dir, source)
    # watch out! u2is must get from target market
    ddata_path = pjoin(data_dir, target)

    ddata = load_graph(ddata_path, merge_valid=False)
    if source == 'all':
        sdata = load_all_graph(data_dir, merge_valid=True, exlude=target)
    else:
        sdata = load_graph(sdata_path)

    if concat:
        sdata = merge_df(sdata, ddata)
    _, u2is = df2dict(ddata)
    i2us, _ = df2dict(sdata)
    lgb_valid_data = cal_i2u_u(lgb_valid_data, u2is, i2us, prefix='{}_{}_i2u'.format(concat_desc, source), sim='iou')
    lgb_valid_data = cal_i2u_u(lgb_valid_data, u2is, i2us, prefix='{}_{}_i2u'.format(concat_desc, source), sim='cosine')

    ddata = load_graph(ddata_path, merge_valid=True)
    if source == 'all':
        sdata = load_all_graph(data_dir, merge_valid=True, exlude=target)
    else:
        sdata = load_graph(sdata_path)

    if concat:
        sdata = merge_df(sdata, ddata)
    _, u2is = df2dict(ddata)
    i2us, _ = df2dict(sdata)
    lgb_test_data = cal_i2u_u(lgb_test_data, u2is, i2us, prefix='{}_{}_i2u'.format(concat_desc, source), sim='cosine')
    lgb_test_data = cal_i2u_u(lgb_test_data, u2is, i2us, prefix='{}_{}_i2u'.format(concat_desc, source), sim='iou')

    return lgb_valid_data, lgb_test_data

def item2vec_score(data, source_markets, target_market, latent_dim, testing=False):
    data_graph = load_multi_graph('../input/', source_markets, merge_train=True, merge_valid=testing)
    item_corpus_list = data_graph.groupby('userId')['itemId'].agg(list).reset_index() 
    sentences_list =[]
    for row in item_corpus_list.itertuples():
        sentences_list.append(row.itemId)

    dim = latent_dim
    model = Word2Vec(sentences_list, 
                    size=dim, 
                    window=40, 
                    min_count=1, 
                    sg=1, 
                    negative=10,
                    hs=0, 
                    # epochs=8, 
                    seed=996)
    vecs_all = model.wv.vectors/np.linalg.norm(model.wv.vectors,axis=1,keepdims=True)
    def get_stats_sim(row):
        target_item = row['itemId']
        item_lst = row['itemIds']
        target_vec  = model.wv[target_item] if target_item in model.wv else 0.
        target_vec = target_vec / (np.linalg.norm(target_vec,keepdims=True)+0.000001)
        simi_lst = []
        for item in item_lst:
            cur_vec = model.wv[item] if item in model.wv else 0.
            cur_vec = cur_vec / (np.linalg.norm(cur_vec,keepdims=True)+0.000001)
            cur_simi = np.dot(target_vec,cur_vec)
            simi_lst.append(cur_simi)
        return np.mean(simi_lst), np.max(simi_lst), np.min(simi_lst), np.std(simi_lst), np.sum(simi_lst), \
            np.median(simi_lst),np.percentile(simi_lst, 20),np.percentile(simi_lst, 70),np.percentile(simi_lst, 5),np.percentile(simi_lst, 95)

    item_corpus_list.columns = ['userId', 'itemIds']
    data = data.merge(item_corpus_list, on='userId', how='left')

    data['stats'] = data[['itemIds','itemId']].swifter.set_npartitions(16).apply(lambda x:get_stats_sim(x),axis=1)
    prefix = '+'.join(source_markets)
    data[f'item2vec{prefix}_{dim}_i2i_max']    = data['stats'].apply(lambda x: x[1])
    # data[f'item2vec{prefix}_{dim}_i2i_min']    = data['stats'].apply(lambda x: x[2])
    data[f'item2vec{prefix}_{dim}_i2i_std']    = data['stats'].apply(lambda x: x[3])
    data[f'item2vec{prefix}_{dim}_i2i_sum']    = data['stats'].apply(lambda x: x[4])
    data[f'item2vec{prefix}_{dim}_i2i_median'] = data['stats'].apply(lambda x: x[5])
    data[f'item2vec{prefix}_{dim}_i2i_percentile_20']   = data['stats'].apply(lambda x: x[6])
    data[f'item2vec{prefix}_{dim}_i2i_percentile_70']   = data['stats'].apply(lambda x: x[7])
    data[f'item2vec{prefix}_{dim}_i2i_percentile_5']   = data['stats'].apply(lambda x: x[8])
    data[f'item2vec{prefix}_{dim}_i2i_percentile_90']   = data['stats'].apply(lambda x: x[9])
    data[f'item2vec{prefix}_{dim}_i2i_mean']   = data['stats'].apply(lambda x: x[0])
    del data['itemIds'],  data['stats']
    return data

def get_item2vec_score(lgb_valid_data, lgb_test_data, source_markets, target_market, latent_dim):
    lgb_valid_data = item2vec_score(lgb_valid_data, source_markets, target_market, latent_dim, testing=False)
    lgb_test_data = item2vec_score(lgb_test_data, source_markets, target_market, latent_dim, testing=True)
    return lgb_valid_data, lgb_test_data

def rank_score_from_source(lgb_valid_data, lgb_test_data, target='t1', source='s1', data_dir='../input/', concat=True):
    
    concat_desc = 'concat' if concat else ''
    sdata_path = pjoin(data_dir, source)
    # watch out! u2is must get from target market
    ddata_path = pjoin(data_dir, target)

    ddata = load_graph(ddata_path, merge_valid=False)
    if source == 'all':
        sdata = load_all_graph(data_dir, merge_train=True, merge_valid=True, exlude=target)
    else:
        sdata = load_graph(sdata_path)

    if concat:
        sdata = merge_df(sdata, ddata)
    _, u2is = df2dict(ddata)
    i2us, _ = df2dict(sdata)
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='{}_{}_u2i'.format(concat_desc, source), sim='iou')
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='{}_{}_u2i'.format(concat_desc, source), sim='cosine')

    ddata = load_graph(ddata_path, merge_valid=True)
    if source == 'all':
        sdata = load_all_graph(data_dir, merge_train=True, merge_valid=True, exlude=target)
    else:
        sdata = load_graph(sdata_path)

    if concat:
        sdata = merge_df(sdata, ddata)
    _, u2is = df2dict(ddata)
    i2us, _ = df2dict(sdata)
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='{}_{}_u2i'.format(concat_desc, source), sim='cosine')
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='{}_{}_u2i'.format(concat_desc, source), sim='iou')

    return lgb_valid_data, lgb_test_data

def rank_score(lgb_valid_data, lgb_test_data, data_dir):
    '''
    '''
    market_name = data_dir.split('/')[-1]
    print('='*40)
    print('rank score for {}'.format(market_name))
    print('='*40)
    data = load_graph(data_dir, merge_valid=False)
    i2us, u2is = df2dict(data)
    sim = SwingRecall(u2is)
    lgb_valid_data = cal_swing(lgb_valid_data, u2is, i2us, sim, prefix='swing')
    lgb_valid_data = cal_i2u_u(lgb_valid_data, u2is, i2us, prefix='i2u', sim='iou')
    lgb_valid_data = cal_i2u_u(lgb_valid_data, u2is, i2us, prefix='i2u', sim='cosine')
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='u2i', sim='iou')
    lgb_valid_data = cal_u2i_i(lgb_valid_data, u2is, i2us, prefix='u2i', sim='cosine')
    
    data = load_graph(data_dir, merge_valid=True)
    i2us, u2is = df2dict(data)
    sim = SwingRecall(u2is)
    lgb_test_data = cal_swing(lgb_test_data, u2is, i2us, sim, prefix='swing')
    lgb_test_data = cal_i2u_u(lgb_test_data, u2is, i2us, prefix='i2u', sim='iou')
    lgb_test_data = cal_i2u_u(lgb_test_data, u2is, i2us, prefix='i2u', sim='cosine')
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='u2i', sim='iou')
    lgb_test_data = cal_u2i_i(lgb_test_data, u2is, i2us, prefix='u2i', sim='cosine')
    
    return lgb_valid_data, lgb_test_data