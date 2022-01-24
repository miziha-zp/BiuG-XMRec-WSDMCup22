'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import joblib
import os
import pandas as pd
from data_utils import *
from os.path import join as pjoin
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=100)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # self.scheduler.step(loss)

        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================


def train2txt(df, txt_path, append='w'):
    print('train2txt...')
    train_view = df.groupby('userId')['itemId'].apply(set).reset_index()
    print(train_view.head())
    with open(txt_path, append) as f:
        for u, items in zip(train_view['userId'], train_view['itemId']):
            items = list(items)
            str_line = str(u) + ' ' + str(items[0])
            for item in items[1:]:
                str_line += ' {}'.format(item)
            str_line += '\n'
            f.write(str_line)

class idsMap:
    def __init__(self, ids=[], name=''):
        print('__init__===>{}'.format(name))
        self.idx2idmap = {}
        self.id2idxmap = {}
        self.name = name
        self.update(ids)
        
    def len(self):
        return len(self.id2idxmap)

    def update(self, ids):
        print('-*-'*20)
        print('{}::::before update map size={}'.format(self.name, self.len()))
        for term in ids:
            if term in self.idx2idmap: pass
            else:
                cur_size = self.len()
                self.idx2idmap[term] = cur_size
                self.id2idxmap[cur_size] = term
        print('{}::::after updatemap size={}'.format(self.name, self.len()))
        print('-*-'*20)

    def idx2id(self, arr):
        return [self.idx2idmap[t]for t in arr]
         
    def id2idx(self, arr):
        return [self.id2idxmap[t]for t in arr]


def create_global_graph(market_list=['s1', 's2', 's3', 't1', 't2'], onlytest=False):
    ans = pd.DataFrame()
    for market in market_list:
        t_train = pd.read_csv(os.path.join('../../input/{}/'.format(market), 'train.tsv'), sep='\t')
        ans = pd.concat([ans, t_train], axis=0)
        t_train = pd.read_csv(os.path.join('../../input/{}/'.format(market), 'train_5core.tsv'), sep='\t')
        ans = pd.concat([ans, t_train], axis=0)
        world.cprint(f'data shape:{ans.shape}')
        if onlytest:
            t_train = pd.read_csv(os.path.join('../../input/{}/'.format(market), 'valid_qrel.tsv'), sep='\t')
            ans = pd.concat([ans, t_train], axis=0)
            world.cprint(f'===>after add valid data shape:{ans.shape}')

    return ans[['userId', 'itemId']]

def create_filter_graph(market_list=['s1', 's2', 's3', 't1', 't2'], f_ilter='t1'):
    ans = pd.DataFrame()
    for market in market_list:
        if market == f_ilter:
            t_train = pd.read_csv(os.path.join('../../../input/{}/'.format(market), 'train.tsv'), sep='\t')
            ans = pd.concat([ans, t_train], axis=0)
            filter_train5core = pd.read_csv(os.path.join('../../../input/{}/'.format(f_ilter), 'train_5core.tsv'), sep='\t')
            filter_items = set(filter_train5core['itemId'])
            ans['bool'] = ans['itemId'].apply(lambda x: x in filter_items)
            ans = ans[ans['bool'] == 1]
            del ans['bool']
        t_train = pd.read_csv(os.path.join('../../../input/{}/'.format(market), 'train_5core.tsv'), sep='\t')
        ans = pd.concat([ans, t_train], axis=0)
        world.cprint(f'data shape:{ans.shape}')

    world.cprint(f'data shape:{ans.shape}')
    return ans[['userId', 'itemId']]

def pretrain_finetune_process(data_dir, domain_name='t1'):
    '''
    # transfer data from to ultraGCN format
    # train5core->txt
    '''
    target_dir = f'../data/{domain_name}/'
    pkl_path = pjoin(target_dir, 'id2idx.pkl')
    train_txt_path = pjoin(target_dir, 'train.txt')
    test_txt_path = pjoin(target_dir, 'test.txt')

    finetunetrain_txt_path = pjoin(target_dir, 'f_train.txt')
    finetunetest_txt_path = pjoin(target_dir, 'f_test.txt')

    # input_path
    # train_data = create_global_graph(['t2', 't1'])
    train_data = create_filter_graph([domain_name, 's1'], '')
    finetune_data = create_filter_graph([domain_name], '')
    
    userIdsMap = idsMap(train_data['userId'], 'userId')
    itemIdsMap = idsMap(train_data['itemId'], 'itemId')
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(data_dir)
    userIdsMap.update(lgb_valid_data['userId'])
    userIdsMap.update(lgb_test_data['userId'])
    itemIdsMap.update(lgb_valid_data['itemId'])
    itemIdsMap.update(lgb_test_data['itemId'])

    train_data['userId'] = userIdsMap.idx2id(train_data['userId'])
    train_data['itemId'] = itemIdsMap.idx2id(train_data['itemId'])
    
    finetune_data['userId'] = userIdsMap.idx2id(finetune_data['userId'])
    finetune_data['itemId'] = itemIdsMap.idx2id(finetune_data['itemId'])
    
    # valid_qrel
    valid_qrel = pd.read_csv(os.path.join(data_dir, 'valid_qrel.tsv'), sep='\t').sort_values('userId')
    valid_qrel['userId'] = userIdsMap.idx2id(valid_qrel['userId'])
    valid_qrel['itemId'] = itemIdsMap.idx2id(valid_qrel['itemId'])
    # write to txt

    train2txt(train_data, train_txt_path)
    train2txt(valid_qrel, test_txt_path)
    train2txt(valid_qrel, finetunetest_txt_path)
    train2txt(finetune_data, finetunetrain_txt_path)
    print('wrote to txt file.')
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(data_dir)
    def to_id(df, userIdsmap, itemIdsmap):
        df['userId'] = userIdsmap.idx2id(df['userId'])
        df['itemId'] = itemIdsmap.idx2id(df['itemId'])
        return df
    
    lgb_valid_data = to_id(lgb_valid_data, userIdsMap, itemIdsMap)
    lgb_test_data = to_id(lgb_test_data, userIdsMap, itemIdsMap)

    id_seriral_pkl_path = pjoin(data_dir, 'id_seriral.pkl')
    joblib.dump([lgb_valid_data, lgb_test_data], id_seriral_pkl_path)
    joblib.dump(itemIdsMap, pjoin(data_dir, "itemIdsMap.pkl"))
    joblib.dump(userIdsMap, pjoin(data_dir, "userIdsMap.pkl"))

def tokenizer(data_dir, domain_name='t1'):
    train_data = create_global_graph([domain_name], True)
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(data_dir)
    userIdsMap = idsMap(lgb_valid_data['userId'], 'userId')
    itemIdsMap = idsMap(lgb_valid_data['itemId'], 'itemId')
    userIdsMap.update(lgb_test_data['userId'])
    itemIdsMap.update(lgb_test_data['itemId'])
    userIdsMap.update(train_data['userId'])
    itemIdsMap.update(train_data['itemId'])
    joblib.dump(itemIdsMap, pjoin(data_dir, "itemIdsMap.pkl"))
    joblib.dump(userIdsMap, pjoin(data_dir, "userIdsMap.pkl"))

def data_process(data_dir, domain_name='t1', training=False):
    '''
    # transfer data from to ultraGCN format
    # train5core->txt
    '''
    if training:
        target_dir = f'../data/{domain_name}/'
    else:
        target_dir = f'../data/{domain_name}_testing/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    pkl_path = pjoin(target_dir, 'id2idx.pkl')
    train_txt_path = pjoin(target_dir, 'train.txt')
    test_txt_path = pjoin(target_dir, 'test.txt')
    # input_path
    testing = not training
    train_data = create_global_graph([domain_name], testing)
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(data_dir)
    itemIdsMap = joblib.load(pjoin(data_dir, "itemIdsMap.pkl"))
    userIdsMap = joblib.load(pjoin(data_dir, "userIdsMap.pkl"))
    
    train_data['userId'] = userIdsMap.idx2id(train_data['userId'])
    train_data['itemId'] = itemIdsMap.idx2id(train_data['itemId'])

    # valid_qrel
    valid_qrel = pd.read_csv(os.path.join(data_dir, 'valid_qrel.tsv'), sep='\t').sort_values('userId')
    valid_qrel['userId'] = userIdsMap.idx2id(valid_qrel['userId'])
    valid_qrel['itemId'] = itemIdsMap.idx2id(valid_qrel['itemId'])
    # write to txt
    train2txt(train_data, train_txt_path)
    train2txt(valid_qrel, test_txt_path)
    print('wrote to txt file.')
    lgb_valid_data, lgb_test_data = get_ranking_raw_data(data_dir)
    def to_id(df, userIdsmap, itemIdsmap):
        df['userId'] = userIdsmap.idx2id(df['userId'])
        df['itemId'] = itemIdsmap.idx2id(df['itemId'])
        return df
    
    lgb_valid_data = to_id(lgb_valid_data, userIdsMap, itemIdsMap)
    lgb_test_data = to_id(lgb_test_data, userIdsMap, itemIdsMap)

    id_seriral_pkl_path = pjoin(data_dir, 'id_seriral.pkl')
    joblib.dump([lgb_valid_data, lgb_test_data], id_seriral_pkl_path)

    
def load_checkpoint(model, model_path):
    print('===> load model from:', model_path)
    model.load_state_dict(torch.load(model_path)) 
    return model
