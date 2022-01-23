import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import gc
import os
import time
from tqdm import tqdm
import math

def score_itemidforuserid(sim, num, item_users, uid, iid, topK=100):
    # 对验证数据中的每个item进行TopN推荐
    # 通过相似度矩阵得到与当前用户最相似的前K个item，
    # 然后查看单前itemid是否出现在这K个用户的历史交互序列里，统计出现的次数，并将similarity*1记录
    cnt = 0
    weighted_cnt = 0
    if iid not in sim: return -1, -1
    for v, score in sorted(sim[iid].items(), key=lambda x: x[1], reverse=True)[:topK]: # 选择与物品u最相思的k个item
        if uid in item_users[v]:
            cnt += 1
            weighted_cnt += score
        
    return cnt, weighted_cnt
    
def score_for_series(sim, num, item_users, data, topK=10, prefix=''):
    cnt_list = []
    weighted_cnt_list = []
    for uid, iid in tqdm(zip(data['userId'], data['itemId'])):
        cnt, weighted_cnt = score_itemidforuserid(sim, num, item_users, uid, iid, topK=topK)
        cnt_list.append(cnt)
        weighted_cnt_list.append(weighted_cnt)
    
    # data[prefix+'_XMitemCFscore_list'] = cnt_list
    data[prefix+'_XMitemCFscore_weighted_list'] = weighted_cnt_list
    return data
# 
def itemCF_score(train_data, lgb_valid_data, lgb_test_data, K=10, prefix='train5core'):
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

    # 计算item协同过滤矩阵
    # 即利用item-users倒排表统计item之间共同被购买的用户数量，用户协同过滤矩阵的表示形式为：sim = {item_id1: {item_id2: num1}, item_id3:{item_id4: num2}, ...}
    # 协同过滤矩阵是一个双层的字典，用来表示item之间共同被购买的用户数量
    # 在计算用户协同过滤矩阵的同时还需要记录每个每个物品被购买的次数，其表示形式为: num = {item_id1：num1, item_id2:num2, ...}
    sim = {}
    num = {}
    print('构建协同过滤矩阵...')
    for user, items in tqdm(trn_user_items.items()): # 遍历所有的item去统计,用户两辆之间共同交互的item数量
        for u in items:
            if u not in num: # 如果用户u不在字典num中，提前给其在字典中初始化为0,否则后面的运算会报key error
                num[u] = 0
            num[u] += 1 # 统计每一个item,交互的总user的数量
            if u not in sim: # 如果用户u不在字典sim中，提前给其在字典中初始化为一个新的字典,否则后面的运算会报key error
                sim[u] = {}
            for v in items:
                if u != v:  # 只有当u不等于v的时候才计算用户之间的相似度　
                    if v not in sim[u]:
                        sim[u][v] = 0
                    sim[u][v] += 1

    # 计算item相似度矩阵
    # item协同过滤矩阵其实相当于是余弦相似度的分子部分,还需要除以分母,即两个item分别被购买的user数量的乘积
    # 两个item分别被购买的user数量的乘积就是上面统计的num字典
    print('计算相似度...')
    for u, items in tqdm(sim.items()):
        for v, score in items.items():
            sim[u][v] =  score / math.sqrt(num[u] * num[v]) # 余弦相似度分母部分 
    
    maxn = 0
    for u in sim:
        maxn = 0
        minn = 1e9
        for v in sim[u]:
            if maxn < sim[u][v]:
                maxn = sim[u][v]
            if minn > sim[u][v]:
                minn  = sim[u][v]
        for v in sim[u]:
            sim[u][v] = (sim[u][v] - minn) / (maxn - minn + 1e-9)

    print('score for valid')
    lgb_valid_data = score_for_series(sim, num, item_users, lgb_valid_data, K, prefix=prefix)
    print('score for test')
    lgb_test_data = score_for_series(sim, num, item_users, lgb_test_data, K, prefix=prefix)
    
    return lgb_valid_data, lgb_test_data

def merge_df(df_5core,df_oral,thed):
    df = df_oral.copy()
    df['rating'] = df['rating'].apply(lambda x:1 if x>= thed else 0)
    df = df.loc[df['rating']==1]
    data = pd.concat([df,df_5core])
    data.drop_duplicates(subset=['userId','itemId'],keep='first',inplace=True)
    return data
    
def get_all_market():
    d_path = '../input/'
    ans = pd.DataFrame()
    '''
    only t2: 0.55
    concat([t1, t2]): 0.567
    '''
    for market_name in ['t1', 't2', 's1']:
        train5core = pd.read_csv(os.path.join(d_path, '{}/train_5core.tsv'.format(market_name)), sep='\t').sort_values('userId')
        train = pd.read_csv(os.path.join(d_path, '{}/train.tsv'.format(market_name)), sep='\t').sort_values('userId')
        t = merge_df(train5core, train, 0)
        ans = pd.concat([ans, t], axis=0)
    return ans 

def get_crossitemCF_score(train, train5core, lgb_valid_data, lgb_test_data, K_list=[100]):
    # wrappers for userCF_score
    print('--'*100)
    # print(train[train['userId'] == 't1U1009799'])
    # print(train5core[train5core['userId'] == 't1U1009799'])
    # print(lgb_valid_data[lgb_valid_data['userId'] == 't1U1009799'])
    allmarket = get_all_market()
    for K in K_list:
        print('-'*20, K, '-'*20)
        lgb_valid_data, lgb_test_data = itemCF_score(allmarket, lgb_valid_data, lgb_test_data,K, prefix='XM@K={}'.format(K))
        # lgb_valid_data, lgb_test_data = itemCF_score(train, lgb_valid_data, lgb_test_data, K, prefix='train@K={}'.format(K))
    return  lgb_valid_data, lgb_test_data