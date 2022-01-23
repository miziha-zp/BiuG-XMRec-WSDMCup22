import pandas as pd 
import argparse
import os
from os.path import join as pjoin
import sys
from data_utils import *
from ultragcn import *
from validate_submission import *
import joblib
import torch

def load_checkpoint(model, model_path):
    print('===> load model from:', model_path)
    model.load_state_dict(torch.load(model_path)) 
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Path to the DATA dir of the kit. Default: ../input/t1", default='../input/t1')
    parser.add_argument('--config-file', type=str, help='config file', default='./t1.ini')
    # parser.add_argument('--model-path', type=str, help='config file', default='./save_model/ultragcn_t1.pt')
    parser.add_argument('--only-test', action='store_true')
    args = parser.parse_args()
    market_name = os.path.split(args.dir)[-1]
    # data_process(args.dir, market_name)
    print('1. Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(args.config_file)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    testing = 'testing' in args.config_file
    ultragcn = ultragcn.to(params['device'])
    if not args.only_test:
        train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params, args.dir, testing)
    else:
        ultragcn = load_checkpoint(ultragcn, params['model_save_path'])
    predict(ultragcn, args.dir, testing)

    print('END')
