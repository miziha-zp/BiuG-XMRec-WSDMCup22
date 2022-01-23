import sys
sys.path.append("../../external/daisyRec/")
sys.path.append("../../../utils/")

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from os.path import join
from datetime import datetime
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
from daisy.utils.metrics import RecallPrecision_ATk, MRRatK_r, NDCGatK_r, HRK_r
from daisy.model.SLiMRecommender import SLIM
from predictor import SLIM_predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Baseline')
    parser.add_argument('--dataset', type=str, default='Gowalla')
    parser.add_argument('--topk', type=str, default='[20, 50]')
    parser.add_argument('--l1', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=1e-2)
    args = parser.parse_args()

    print(args)

    # load data
    TRAIN_PATH = join('../../../LightGCN-PyTorch-master/data/' + args.dataset + '/', 'train.txt')
    TEST_PATH = join('../../../LightGCN-PyTorch-master/data/' + args.dataset + '/', 'test.txt')

    # item popularities
    with open(TRAIN_PATH, 'r') as f:
        train_data = f.readlines()
    
    train_set = {'user':[], 'item':[], 'rating':[]}
    user_num, item_num = 0, 0
    item_set = set()
    interacted_items = {}
    for i, line in enumerate(train_data):
        line = line.strip().split(' ')
        user = int(line[0])
        interacted_items[user] = set()
        for iid in line[1:]:
            iid = int(iid)
            train_set['user'].append(user)
            train_set['item'].append(iid)
            train_set['rating'].append(1.0)

            if iid not in item_set:
                item_num += 1
                item_set.add(iid)
            interacted_items[user].add(iid)

    user_num = max(train_set['user']) + 1
    item_num = max(train_set['item']) + 1
    
    train_set = pd.DataFrame(train_set)
    print(train_set.head())
    model = SLIM(user_num, item_num, l1_ratio=args.l1, alpha=args.alpha)

    print('model fitting...')
    model.fit(train_set)

    print('Generate recommend list...')

    
    with open(TEST_PATH, 'r') as f:
        test_data = f.readlines()
    
    topks = eval(args.topk)
    max_k = max(topks)
    
    r = np.zeros((len(test_data), max_k))
    ground_truth = []
    hits = 0
    testing = 'test' in args.dataset
    SLIM_predict(model, '../../../input/{}'.format(args.dataset.split('_')[0]), testing=testing)