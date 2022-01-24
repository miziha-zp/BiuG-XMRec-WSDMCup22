import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import gc
import os
import time
from tqdm import tqdm
import math
import joblib
from termcolor import colored
def load_Recall_score(data_dir, fn, lgb_valid_data, lgb_test_data, training=False):
    print(colored(f'load from {fn}', 'red'))
    features_ultraGCN_pkl_path = os.path.join(data_dir, f'{fn}.pkl')
    features_ultraGCN_pkl = joblib.load(features_ultraGCN_pkl_path)
    features_ultraGCN_valid, features_ultraGCN_test = features_ultraGCN_pkl[0], features_ultraGCN_pkl[1]
    del features_ultraGCN_valid['score'], features_ultraGCN_test['score']
    if training:
        data = lgb_valid_data.merge(features_ultraGCN_valid, on=['userId','itemId'], how='left')
    else:
        data = lgb_test_data.merge(features_ultraGCN_test, on=['userId','itemId'], how='left')
    return data

def load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway=''):
    fn = f'testing_{recallway}'
    lgb_test_data = load_Recall_score(data_dir, fn, lgb_valid_data, lgb_test_data, training=False)
    fn = recallway
    lgb_valid_data = load_Recall_score(data_dir, fn, lgb_valid_data, lgb_test_data, training=True)
    return lgb_valid_data, lgb_test_data
    
def load_lightgcn(data_dir, lgb_valid_data, lgb_test_data):
    lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='lgcn_score_score')
    return lgb_valid_data, lgb_test_data

def load_mf(data_dir, lgb_valid_data, lgb_test_data):
    lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='mf_score_score')
    return lgb_valid_data, lgb_test_data

def load_lgbide(data_dir, lgb_valid_data, lgb_test_data):
    lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='lgn-ide_score')
    return lgb_valid_data, lgb_test_data

def load_gfcf(data_dir, lgb_valid_data, lgb_test_data):
    lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='gf-cf_score')
    return lgb_valid_data, lgb_test_data

def load_slim(data_dir, lgb_valid_data, lgb_test_data):
    lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='SLIM_scores_score')
    return lgb_valid_data, lgb_test_data   

def add_utils(data, Klist=[1, 2, 3], test=False):
    data[f'sum_itemKNN{Klist[0]}'] = 0
    for k in Klist:
        data[f'sum_itemKNN{Klist[0]}'] += data['itemKNN{}_scores'.format(k)]
    return data

def load_itemEASE_r(data_dir, lgb_valid_data, lgb_test_data):
    for l2 in [700.0]:
        desp = str(l2)
        final_name = f'EASE_r{desp}'
        lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway=f'{final_name}_scores_score')
    return lgb_valid_data, lgb_test_data   

def load_itemKNN(data_dir, lgb_valid_data, lgb_test_data):
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:#  
        lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='itemKNN{}_scores_score'.format(k))

    lgb_valid_data = add_utils(lgb_valid_data)
    lgb_test_data = add_utils(lgb_test_data)
    
    # lgb_valid_data = add_utils(lgb_valid_data,[5,7,9])
    # lgb_test_data = add_utils(lgb_test_data,[5,7,9])


    # lgb_valid_data = add_utils(lgb_valid_data,[8, 9, 10, 15, 20])
    # lgb_test_data = add_utils(lgb_test_data,[8, 9, 10, 15, 20])

    # lgb_valid_data = add_utils(lgb_valid_data,[10, 15, 20])
    # lgb_test_data = add_utils(lgb_test_data,[10, 15, 20])
    
    return lgb_valid_data, lgb_test_data  

def load_ultragcn(data_dir, lgb_valid_data, lgb_test_data):
    lgb_valid_data, lgb_test_data = load_Recall_score_split(data_dir, lgb_valid_data, lgb_test_data, recallway='features_ultraGCN_score')
    return lgb_valid_data, lgb_test_data


def load_item2vec(data_dir, lgb_valid_data, lgb_test_data):
    features_ultraGCN_pkl_path = os.path.join(data_dir, 'item2vec_scores1024.pkl')
    features_ultraGCN_pkl = joblib.load(features_ultraGCN_pkl_path)
    features_ultraGCN_valid, features_ultraGCN_test = features_ultraGCN_pkl[0], features_ultraGCN_pkl[1]
    del features_ultraGCN_valid['score'], features_ultraGCN_test['score']
    lgb_valid_data = lgb_valid_data.merge(features_ultraGCN_valid, on=['userId','itemId'], how='left')
    lgb_test_data = lgb_test_data.merge(features_ultraGCN_test, on=['userId','itemId'], how='left')
    return lgb_valid_data, lgb_test_data




# def load_gfcf(data_dir, lgb_valid_data, lgb_test_data):
#     features_ultraGCN_pkl_path = os.path.join(data_dir, 'gf-cf_score.pkl')
#     features_ultraGCN_pkl = joblib.load(features_ultraGCN_pkl_path)
#     features_ultraGCN_valid, features_ultraGCN_test = features_ultraGCN_pkl[0], features_ultraGCN_pkl[1]
#     del features_ultraGCN_valid['score'], features_ultraGCN_test['score']
#     lgb_valid_data = lgb_valid_data.merge(features_ultraGCN_valid, on=['userId','itemId'], how='left')
#     lgb_test_data = lgb_test_data.merge(features_ultraGCN_test, on=['userId','itemId'], how='left')
#     return lgb_valid_data, lgb_test_data



# def load_itemKNN(data_dir, lgb_valid_data, lgb_test_data):
#     for k in [1, 2, 3, 4, 5, 7, 9, 10, 15]:
#         features_ultraGCN_pkl_path = os.path.join(data_dir, 'itemKNN_scores{}.pkl'.format(k))
#         features_ultraGCN_pkl = joblib.load(features_ultraGCN_pkl_path)
#         features_ultraGCN_valid, features_ultraGCN_test = features_ultraGCN_pkl[0], features_ultraGCN_pkl[1]
#         del features_ultraGCN_valid['score'], features_ultraGCN_test['score']
#         lgb_valid_data = lgb_valid_data.merge(features_ultraGCN_valid, on=['userId','itemId'], how='left')
#         lgb_test_data = lgb_test_data.merge(features_ultraGCN_test, on=['userId','itemId'], how='left')
#     return lgb_valid_data, lgb_test_data


def load_global_features(data_dir, lgb_valid_data, lgb_test_data):
    global_item_count = get_global_item_count()
    lgb_valid_data = lgb_valid_data.merge(global_item_count, on='itemId', how='left')
    lgb_test_data = lgb_test_data.merge(global_item_count, on='itemId', how='left')
    return lgb_valid_data, lgb_test_data
    
def get_global_item_count():
    # check items in s1 is or not cover all other items in other market

    s1train = pd.read_csv(os.path.join('../input/s1/', 'train.tsv'), sep='\t')
    s1train5core = pd.read_csv(os.path.join('../input/s1/', 'train_5core.tsv'), sep='\t')

    global_count = s1train.groupby('itemId',as_index=False)['userId'].count().rename(columns={"userId":"s1train_count"})
    global_count_ = s1train5core.groupby('itemId',as_index=False)['userId'].count().rename(columns={"userId":"s1train_5core_count"})
    global_count = global_count.merge(global_count_, how='outer', on='itemId').fillna(0)
    
    s1_train_items = set(s1train['itemId']).union(set(s1train5core['itemId']))
    # print(global_count.head())

    print('market s1 :', len(s1_train_items))
    for market in ['s2', 's3', 't1', 't2']:
        t_train = pd.read_csv(os.path.join('../input/{}/'.format(market), 'train.tsv'), sep='\t')
        global_count_ = t_train.groupby('itemId',as_index=False)['userId'].count().rename(columns=     {"userId":"{}train_count".format(market)})
        global_count = global_count.merge(global_count_, how='outer', on='itemId').fillna(0)
        
        t_train = pd.read_csv(os.path.join('../input/{}/'.format(market), 'train_5core.tsv'), sep='\t')
        global_count_ = t_train.groupby('itemId',as_index=False)['userId'].count().rename(columns=     {"userId":"{}train_5core_count".format(market)})
        global_count = global_count.merge(global_count_, how='outer', on='itemId').fillna(0)
        
        t_items = set(t_train['itemId'])
        print('market {} items not in s1: {} / {}'.format(market, len(t_items-s1_train_items), len(t_items)))
    global_count['global_count'] = 0
    for mar in ['s1', 's2', 's3', 't1', 't2']:
        global_count['global_count'] += global_count["{}train_5core_count".format(mar)]

    return global_count

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    
    return df

def groupRank(df, groupname, features_list=['ultraGCNscore']):
    new_featureslist = [fea+'_grouprank' for fea in features_list]
    tmp_data = df[[groupname]+features_list]
    # print(tmp_data)
    groupbysort_data = tmp_data.groupby(groupname).rank(method='min',ascending=False)
    groupbysort_data.columns = new_featureslist
    print(new_featureslist)
    for fea in new_featureslist:
        df[fea] = groupbysort_data[fea]
    return df

def get_feature_here(lgb_valid_data, lgb_test_data):
    def here_fun(df):
        df['here_itemIdcount'] = df['itemId'].map(dict(df['itemId'].value_counts()))
        score_list = []
        df1 = df.copy()
        df = df.set_index('itemId')
        for col in df.columns:
            if 'userCF' in col:score_list.append(col)
        for col in df.columns:
            if 'itemCF' in col:score_list.append(col)    
        for col in df.columns:
            if 'ultraGCN' in col:score_list.append(col)
        for col in df.columns:
            if 'u2i' in col:score_list.append(col)
        for col in df.columns:
            if '_score' in col:score_list.append(col)
            
        # print(score_list)
        for col in score_list:
            df_view = df1.groupby('itemId')[col].mean()
            df['here_itemId_{}_mean'.format(col)] = df_view
            # 
            df['here_{}-itemId_{}_mean'.format(col, col)] = df['here_itemId_{}_mean'.format(col)]-df[col]

        return df.reset_index()
    lgb_valid_data = here_fun(lgb_valid_data)
    lgb_test_data = here_fun(lgb_test_data)

    return lgb_valid_data, lgb_test_data

def emb(df, f1, f2, emb_size = 8):
   
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=50, min_count=5, sg=1, hs=1, seed=2019)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)

    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    tmp = reduce_mem(tmp)

    return tmp

def norm_feature(lgb_valid_data, lgb_test_data):
    for fea in lgb_test_data.columns:
        if 'i2ucosine' in fea:
            if 'train5core' not in fea:
                new_feature_name = 'norm_{}'.format(fea)
                lgb_test_data[new_feature_name] = lgb_test_data[fea] / (lgb_test_data['itemId_count']+0.1)
                lgb_valid_data[new_feature_name] = lgb_valid_data[fea] / (lgb_valid_data['itemId_count']+0.1)
            else:
                new_feature_name = 'norm_{}'.format(fea)
                lgb_test_data[new_feature_name] = lgb_test_data[fea] / (lgb_test_data['train5core_itemId_count']+0.1)
                lgb_valid_data[new_feature_name] = lgb_valid_data[fea] / (lgb_valid_data['train5core_itemId_count']+0.1)
            
    for fea in lgb_test_data.columns:
        if 'u2icosine' in fea:
            if 'train5core' not in fea:
                new_feature_name = 'norm_{}'.format(fea)
                lgb_test_data[new_feature_name] = lgb_test_data[fea] / (lgb_test_data['userId_count']+0.1)
                lgb_valid_data[new_feature_name] = lgb_valid_data[fea] / (lgb_valid_data['userId_count']+0.1)
            else:
                new_feature_name = 'norm_{}'.format(fea)
                lgb_test_data[new_feature_name] = lgb_test_data[fea] / (lgb_test_data['train5core_userId_count']+0.1)
                lgb_valid_data[new_feature_name] = lgb_valid_data[fea] / (lgb_valid_data['train5core_userId_count']+0.1)

    for fea in lgb_test_data.columns:
        if 'swing' in fea:
            if 'train5core' not in fea:
                new_feature_name = 'norm_{}'.format(fea)
                lgb_test_data[new_feature_name] = lgb_test_data[fea] / (lgb_test_data['userId_count']+0.1)
                lgb_valid_data[new_feature_name] = lgb_valid_data[fea] / (lgb_valid_data['userId_count']+0.1)
            else:
                new_feature_name = 'norm_{}'.format(fea)
                lgb_test_data[new_feature_name] = lgb_test_data[fea] / (lgb_test_data['train5core_userId_count']+0.1)
                lgb_valid_data[new_feature_name] = lgb_valid_data[fea] / (lgb_valid_data['train5core_userId_count']+0.1)
        
    return lgb_valid_data, lgb_test_data

def get_w2v_features(train, train5core, lgb_valid_data, lgb_test_data, emb_sz=8):
    emb_cols = [
        ['userId', 'itemId'],
    ]
    for f1, f2 in emb_cols:
        emb1 = emb(train, f1, f2, emb_sz)
        emb2 = emb(train, f2, f1, emb_sz)
        
        lgb_valid_data = lgb_valid_data.merge(emb1, on=f1, how='left').fillna(0)
        lgb_valid_data = lgb_valid_data.merge(emb2, on=f2, how='left').fillna(0)

        lgb_test_data = lgb_test_data.merge(emb1, on=f1, how='left').fillna(0)
        lgb_test_data = lgb_test_data.merge(emb2, on=f2, how='left').fillna(0)    
    gc.collect()
    return lgb_valid_data, lgb_test_data

def score_itemidforuserid(sim, num, trn_user_items, uid, iid, topK=10):
    # 对验证数据中的每个用户进行TopN推荐
    # 在对用户进行推荐之前需要先通过相似度矩阵得到与当前用户最相思的前K个用户，
    # 然后查看单前itemid是否出现在这K个用户的历史交互序列里，统计出现的次数，并将similarity*1记录
    cnt = 0
    weighted_cnt = 0
    if uid not in sim: return -1, -1
    for v, score in sorted(sim[uid].items(), key=lambda x: x[1], reverse=True)[:topK]: # 选择与用户u最相思的k个用户
        if iid in trn_user_items[v]:
            cnt += 1
            weighted_cnt += score
        
    return cnt, weighted_cnt
    
def score_for_series(sim, num, trn_user_items, data, topK=10, prefix=''):
    cnt_list = []
    weighted_cnt_list = []
    for uid, iid in tqdm(zip(data['userId'], data['itemId'])):
        cnt, weighted_cnt = score_itemidforuserid(sim, num, trn_user_items, uid, iid, topK=topK)
        cnt_list.append(cnt)
        weighted_cnt_list.append(weighted_cnt)
    
    # data[prefix+'_userCFscore_list'] = cnt_list
    data[prefix+'_userCFscore_weighted_list'] = weighted_cnt_list
    return data

def userCF_score(train_data, lgb_valid_data, lgb_test_data, K=10, prefix='train5core'):
    # adapt from https://github.com/datawhalechina/fun-rec/blob/master/codes/base_models/UserCF.py
    # score every (userId, itemId) pair in lgb_valid_data and lgb_test_data
    # userCF score：conside the topK similar to userId, and review their opinions on this itemId
    
    trn_data = train_data.groupby('userId')['itemId'].apply(list).reset_index()
    trn_user_items = {}
    # 将数组构造成字典的形式{user_id: [item_id1, item_id2,...,item_idn]}
    for user, movies in zip(*(list(trn_data['userId']), list(trn_data['itemId']))):
        trn_user_items[user] = set(movies)

    # 建立item->users倒排表
    # 倒排表的格式为: {item_id1: {user_id1, user_id2, ... , user_idn}, item_id2: ...} 也就是每个item对应有那些用户有过点击
    # 建立倒排表的目的就是为了更好的统计用户之间共同交互的商品数量
    print('建立倒排表...')
    item_users = {}
    for uid, items in tqdm(trn_user_items.items()): # 遍历每一个用户的数据,其中包含了该用户所有交互的item
        for item in items: # 遍历该用户的所有item, 给这些item对应的用户列表添加对应的uid
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(uid)

    # 计算用户协同过滤矩阵
    # 即利用item-users倒排表统计用户之间交互的商品数量，用户协同过滤矩阵的表示形式为：sim = {user_id1: {user_id2: num1}, user_id3:{user_id4: num2}, ...}
    # 协同过滤矩阵是一个双层的字典，用来表示用户之间共同交互的商品数量
    # 在计算用户协同过滤矩阵的同时还需要记录每个用户所交互的商品数量，其表示形式为: num = {user_id1：num1, user_id2:num2, ...}
    sim = {}
    num = {}
    print('构建协同过滤矩阵...')
    for item, users in tqdm(item_users.items()): # 遍历所有的item去统计,用户两辆之间共同交互的item数量
        for u in users:
            if u not in num: # 如果用户u不在字典num中，提前给其在字典中初始化为0,否则后面的运算会报key error
                num[u] = 0
            num[u] += 1 # 统计每一个用户,交互的总的item的数量
            if u not in sim: # 如果用户u不在字典sim中，提前给其在字典中初始化为一个新的字典,否则后面的运算会报key error
                sim[u] = {}
            for v in users:
                if u != v:  # 只有当u不等于v的时候才计算用户之间的相似度　
                    if v not in sim[u]:
                        sim[u][v] = 0
                    sim[u][v] += 1
    # 计算用户相似度矩阵
    # 用户协同过滤矩阵其实相当于是余弦相似度的分子部分,还需要除以分母,即两个用户分别交互的item数量的乘积
    # 两个用户分别交互的item数量的乘积就是上面统计的num字典
    print('计算相似度...')
    for u, users in tqdm(sim.items()):
        for v, score in users.items():
            sim[u][v] =  score / math.sqrt(num[u] * num[v]) # 余弦相似度分母部分 
    
    
    print('score for valid')
    lgb_valid_data = score_for_series(sim, num, trn_user_items, lgb_valid_data, K, prefix=prefix)
    print('score for test')
    lgb_test_data = score_for_series(sim, num, trn_user_items, lgb_test_data, K, prefix=prefix)
    
    return lgb_valid_data, lgb_test_data

def get_userCF_score(train, train5core, lgb_valid_data, lgb_test_data, K_list=[100]):
    # wrappers for userCF_score
    print('--'*100)
    # print(train[train['userId'] == 't1U1009799'])
    # print(train5core[train5core['userId'] == 't1U1009799'])
    # print(lgb_valid_data[lgb_valid_data['userId'] == 't1U1009799'])
    def merge_df(df_5core,df_oral,thed=0):
        df = df_oral.copy()
        df['rating'] = df['rating'].apply(lambda x:1 if x>= thed else 0)
        df = df.loc[df['rating']==1]
        data = pd.concat([df,df_5core])
        data.drop_duplicates(subset=['userId','itemId'],keep='first',inplace=True)
        return data

    for K in K_list:
        print('-'*20, K, '-'*20)
        # lgb_valid_data, lgb_test_data = userCF_score(train5core, lgb_valid_data, lgb_test_data,K, prefix='train5core@K={}'.format(K))
        # lgb_valid_data, lgb_test_data = userCF_score(train, lgb_valid_data, lgb_test_data, K, prefix='train@K={}'.format(K))
        train_data = merge_df(train5core, train)
        lgb_valid_data, lgb_test_data = userCF_score(train_data, lgb_valid_data, lgb_test_data, K, prefix='concattrain@K={}'.format(K))
        
    return  lgb_valid_data, lgb_test_data

def stats(arr):
    arr = np.array(arr)
    return min(arr), max(arr), np.mean(arr), np.std(arr), np.median(arr), len(arr)

def hist_rating(arr):
    ans = [0] * 5
    arr_t = [int(t) for t in arr]
    for t in arr_t:
        assert t >= 1 and t <= 5
        ans[t-1] += 1
    return ans

def get_train_features(train, train5core, lgb_valid_data, lgb_test_data):
    print('get_train_features')
    # print(lgb_valid_data.head())
    # for train
    train_item_features = train.groupby('itemId')['rating'].apply(list).reset_index(drop=False)
    # train_item_features['itemId_rating_mean'] = train_item_features['rating'].apply(np.mean)
    # train_item_features['itemId_rating_std'] = train_item_features['rating'].apply(np.std)
    # train_item_features['itemId_rating_max'] = train_item_features['rating'].apply(np.max)
    # train_item_features['itemId_rating_std'] = train_item_features['rating'].apply(np.std)
    # train_item_features['itemId_rating_min'] = train_item_features['rating'].apply(np.min)
    train_item_features['itemId_count'] = train_item_features['rating'].apply(len)

    train_item_features['rating_count'] = train_item_features['rating'].apply(hist_rating)
    for i in range(5):
        train_item_features['item_rating={}_count'.format(i+1)] = train_item_features['rating_count'].apply(lambda x: x[i])
    del train_item_features['rating'], train_item_features['rating_count']

    lgb_valid_data = lgb_valid_data.merge(train_item_features, on='itemId', how='left').fillna(-1)
    lgb_test_data = lgb_test_data.merge(train_item_features, on='itemId', how='left').fillna(-1)
    
    train_user_features = train.groupby('userId')['rating'].apply(list).reset_index(drop=False)
    # train_user_features['userId_rating_mean'] = train_user_features['rating'].apply(np.mean)
    # train_user_features['userId_rating_std'] = train_user_features['rating'].apply(np.std)
    # train_user_features['userId_rating_max'] = train_user_features['rating'].apply(np.max)
    # train_user_features['userId_rating_std'] = train_user_features['rating'].apply(np.std)
    # train_user_features['userId_rating_min'] = train_user_features['rating'].apply(np.min)
    train_user_features['userId_count'] = train_user_features['rating'].apply(len)

    # train_user_features['rating_count'] = train_user_features['rating'].apply(hist_rating)
    # for i in range(5):
    #     train_user_features['userId_rating={}_count'.format(i+1)] = train_user_features['rating_count'].apply(lambda x: x[i])
    # del train_user_features['rating'], train_user_features['rating_count']
    
    lgb_valid_data = lgb_valid_data.merge(train_user_features, on='userId', how='left').fillna(-1)
    lgb_test_data = lgb_test_data.merge(train_user_features, on='userId', how='left').fillna(-1)
    # print('user count max: count', lgb_valid_data['userId_count'].max()) 
    # print('item count max: count', lgb_valid_data['itemId_count'].max()) 
    print('-'*100)
    # for train5core
    train5core_feature_userId = train5core.groupby('userId')['rating'].count().to_dict()
    train5core_feature_itemId = train5core.groupby('itemId')['rating'].count().to_dict()
    
    lgb_valid_data['train5core_userId_count'] = lgb_valid_data['userId'].map(train5core_feature_userId).fillna(-1)
    lgb_valid_data['train5core_itemId_count'] = lgb_valid_data['itemId'].map(train5core_feature_itemId).fillna(-1)

    lgb_test_data['train5core_itemId_count'] = lgb_test_data['itemId'].map(train5core_feature_itemId).fillna(-1)
    lgb_test_data['train5core_userId_count'] = lgb_test_data['userId'].map(train5core_feature_userId).fillna(-1)

    # print(lgb_test_data.describe())
    return lgb_valid_data, lgb_test_data

if __name__ == '__main__':
    arr = [1,3,4,5,5,2]
    print(hist_rating(arr))
    print(stats(arr))
    