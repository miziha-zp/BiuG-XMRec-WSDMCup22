#======================================================================
# EASE^r model for implicit CF, proposed by the following paper:
# + [WWW'2019] Embarrassingly Shallow Autoencoders for Sparse Data
# Authors: XUEPAI Team
#======================================================================

from EASE_rec import EASE
import argparse
from datetime import datetime

import sys
sys.path.append("../../../utils/")
from predictor import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True,
                        help='Training data path')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Testing data path')
    parser.add_argument('--l2_reg', type=float, required=True,
                        help='regularization weight')
    parser.add_argument('--dataset', type=str, default='t1',
                        help=' dataset')
                        
    args = parser.parse_args()

    print("{} {}".format(datetime.now(), args))
    model = EASE()
    model.fit(args.train_data, args.l2_reg)
    metrics = ["Recall(k=20)", "Recall(k=50)", "NDCG(k=20)", "NDCG(k=50)", "HitRate(k=20)", "HitRate(k=50)"]
    result = model.evaluate(args.test_data, metrics)
    # with open("./experimental_result.csv", "a+") as fout:
    #     print(datetime.now(), args.train_data, args.test_data, args.l2_reg, result, sep=",", file=fout)
    testing = 'test' in args.dataset
    EASE_r_predict(model, '../../../input/{}'.format(args.dataset.split('_')[0]), testing=testing,desp=str(args.l2_reg))