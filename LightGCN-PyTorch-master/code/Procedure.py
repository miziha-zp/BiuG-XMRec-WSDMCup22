'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import os
import joblib
from os.path import join as pjoin
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from validate_submission import *


CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def save(ids_valid_data, ids_test_data, data_path, description='test',testing=False):
    testing_des = 'testing_' if testing else ''
    pkl_path = pjoin(data_path, f'{testing_des}{description}_score.pkl')
    print('saving ====> {}'.format(pkl_path))
    joblib.dump([ids_valid_data, ids_test_data], pkl_path)


def predict(model, data_path, epoch, seed='2020', testing=False, description=''):
    market_name = os.path.split(data_path)[-1]
    print('===> market: {}'.format(market_name))
    print('===> predict:')
    id_seriral_pkl_path = pjoin(data_path, 'id_seriral.pkl')
    pkl_arr = joblib.load(id_seriral_pkl_path)
    ids_valid_data, ids_test_data = pkl_arr
    # predict 
    def predict_dot_product_score(df, model, description=''):
        # TODO: device
        users = torch.from_numpy(df['userId'].values).to('cuda:0')
        items = torch.from_numpy(df['itemId'].values).to('cuda:0')
        # print()
        model.eval()
        test_batch_s = 20480
        batch_number = len(users) // test_batch_s + 1 if len(users) % test_batch_s else 0
        scores , uembs, iembs = [], [], []
        with torch.no_grad():
            for i in range(batch_number):
                score, uemb, iemb = model.xmrec_test(users[i*test_batch_s:(i+1)*test_batch_s], \
                    items[i*test_batch_s:(i+1)*test_batch_s])
                scores.append(score)
                uembs.append(uemb)
                iembs.append(iemb)

        
        scores = torch.cat(scores, dim=0)
        df[f'{description}_score'] = scores.squeeze().detach().cpu().numpy()

        return df

    ids_valid_data = predict_dot_product_score(ids_valid_data, model, description=description)
    ids_test_data = predict_dot_product_score(ids_test_data, model, description=description)
    
    # reverse_ids
    def return_idx(df, itemIdsmap, userIdsmap):
        df['userId'] = userIdsmap.id2idx(df['userId'])
        df['itemId'] = itemIdsmap.id2idx(df['itemId'])
        return df

    print('====> load IdsMap')
    itemIdsMap = joblib.load(pjoin(data_path, "itemIdsMap.pkl"))
    userIdsMap = joblib.load(pjoin(data_path, "userIdsMap.pkl"))
    ids_valid_data = return_idx(ids_valid_data, itemIdsMap, userIdsMap)
    ids_test_data = return_idx(ids_test_data, itemIdsMap, userIdsMap)
    # cal score from groudtruth.
    offline_scores(ids_valid_data, f'{description}_score',data_path, market_name)
    # save score as features for lightgbm
    return ids_valid_data, ids_test_data

            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
