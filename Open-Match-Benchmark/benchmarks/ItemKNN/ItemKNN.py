#======================================================================
# ItemKNN model for implicit CF, modified based on the following papers:
# + [WWW'2001] Item-Based Collaborative Filtering Recommendation Algorithms
# + [SIGIR'2007] Effective Missing Data Prediction for Collaborative Filtering
# Authors: Jinpeng Wang <Tsinghua University>
#          Kelong Mao <Tsinghua University>
# Edited by XUEPAI Team
#======================================================================

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from ItemKNN_utils import evaluate_metrics
import os
import sys
from os.path import join
sys.path.append("../../../utils/")
from predictor import *

class ItemKNN:
    def __init__(self, params):
        print("%s Params: %s" % (datetime.now(), params))
        self._readDataset(params.get("train_data"), params.get("test_data"))
        self.similarity_measure = params.get("similarity_measure")
        self.num_neighbors = params.get("num_neighbors")
        self.renormalize_similarity = params.get("renormalize_similarity")
        self.enable_average_bias = params.get("enable_average_bias")
        self.metrics = params.get("metrics")
        self.min_similarity_threshold = params.get("min_similarity_threshold")
        self.pred_matrix = 0
    def _readDataset(self, train_data, test_data):
        print("%s Reading dataset..." % datetime.now())
        train = pd.read_csv(train_data, sep="\t").values.astype(int)
        test = pd.read_csv(test_data, sep="\t").values.astype(int)
        total = np.vstack([train, test])
        self.nUsers = np.max(total[:, 0]) + 1
        self.nItems = np.max(total[:, 1]) + 1
        print("%s Number of users: %d, number of items: %d" % (datetime.now(),  self.nUsers, self.nItems))
        self.trainIUMatrix = csc_matrix((np.ones(len(train)), train.T), 
                                        shape=(self.nUsers, self.nItems)).T # row:item, col:user
        self.test_items = defaultdict(list)
        for row in test:
            self.test_items[row[0]].append(row[1])

    def fit(self):
        self.sim_matrix = self.get_pairwise_similarity()
        self.sim_matrix[np.isnan(self.sim_matrix)] = -1
        if self.renormalize_similarity:
            self.sim_matrix = (self.sim_matrix + 1) / 2 # map to 0 ~ 1
        self.sim_matrix[self.sim_matrix < self.min_similarity_threshold] = 0 # remove similar values less than threshold
        item_indexes = np.argpartition(-self.sim_matrix, self.num_neighbors)[:, self.num_neighbors:] # pick the smallest
        self.sim_matrix[np.arange(item_indexes.shape[0])[:, np.newaxis], item_indexes] = 0
        self.sim_matrix = normalize(self.sim_matrix, norm='l1', axis=1)
        print('%s Finished similarity matrix computation.' % datetime.now())

    def predict(self):
        print('%s Start predicting preference...' % datetime.now())
        trainIUMatrix = self.trainIUMatrix.toarray()
        if self.enable_average_bias:
            item_mean = np.mean(trainIUMatrix, axis=1, keepdims=True)
            pred_matrix = np.dot(self.sim_matrix, trainIUMatrix - item_mean) + item_mean
        else:
            pred_matrix = np.dot(self.sim_matrix, trainIUMatrix)
        pred_matrix[trainIUMatrix > 0] = -np.inf # remove clicked items in train data
        self.pred_matrix = pred_matrix.T
        return pred_matrix.T

    def score(self, u, i):
        try:
            score_ = self.pred_matrix[u, i]
        except:
            score_ = 0
        return score_
        

    def evaluate(self):
        '''compute hitrate, recall, NDCG @ topK'''
        evaluate_metrics(self.predict(), self.test_items, self.metrics)

    def get_pairwise_similarity(self):
        print('%s Start computing similarity matrix...' % datetime.now())
        if self.similarity_measure == 'pearson':
            return np.corrcoef(self.trainIUMatrix.toarray()) - 2 * np.eye(self.nItems) # set diagnal to -1
        elif self.similarity_measure == 'cosine':
            return cosine_similarity(self.trainIUMatrix.toarray()) - 2 * np.eye(self.nItems) # set diagnal to -1
        else:
            raise NotImplementedError("similarity_measure=%s is not supported." % self.similarity_measure)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple Baseline')
    parser.add_argument('--dataset', type=str, default='t3')
    parser.add_argument('--K', type=int, default=2)
    
    args = parser.parse_args()

    TRAIN_PATH = join('data/data/' + args.dataset + '/', 'train_enmf.txt')
    TEST_PATH = join('data/data/' + args.dataset + '/', 'test_enmf.txt')

    params = {"train_data":TRAIN_PATH,
              "test_data":TEST_PATH,
              "similarity_measure": "pearson",
              "num_neighbors": args.K,
              "min_similarity_threshold": 0.1,
              "renormalize_similarity": True,
              "enable_average_bias": True,
              "metrics": ["F1(k=20)", "Recall(k=20)", "Recall(k=50)", "NDCG(k=20)", "NDCG(k=50)", "HitRate(k=20)", "HitRate(k=50)"]}
    model = ItemKNN(params)
    model.fit()
    model.predict()
    # model.evaluate()
    print('===============>{}'.format(params['num_neighbors']))
    testing = 'test' in args.dataset
    itemKNN_predict(model, '../../../input/{}'.format(args.dataset.split('_')[0]), K=params['num_neighbors'], testing=testing)