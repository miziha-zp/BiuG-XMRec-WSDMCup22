import pandas as pd
import os

def get_gt_dict(gt_file_name):
    valid_qrel = pd.read_csv(gt_file_name, sep='\t').sort_values('userId')
    return dict(zip(valid_qrel['userId'], valid_qrel['itemId']))

def get_valid_file(valid_file_name, gt_file_name):
    '''
    return valid_file in lgb_train_data format and label the data with gt file.
    '''
    gt_dict = get_gt_dict(gt_file_name)
    print('proccessing ', valid_file_name)
    t_data = pd.read_csv(valid_file_name, sep='\t', header=None)
    print('oral data shape:', t_data.shape)
    # transfer -> userid, itemid, score
    userid_list = []
    itemid_list = []
    ans_list = []
    for uid, iid_list in zip(t_data[0], t_data[1]):
        iid_list_ = iid_list.split(',')
        assert len(iid_list_) == 100
        userid_list += [uid] * 100
        itemid_list += iid_list_
        ans_list += [int(gt_dict[uid] == iid)for iid in iid_list_]
    data = pd.DataFrame({
        # userId itemId	score
        "userId": userid_list,
        "itemId": itemid_list,
        "score" : ans_list
    })
    print('transformed data shape:', data.shape)
    print("data['score'].mean():", data['score'].mean())
    return data

def get_test_file(file_name, gt=None):
    print('proccessing ', file_name)
    t_data = pd.read_csv(file_name, sep='\t', header=None)
    print('oral data shape:', t_data.shape)
    # transfer -> userid, itemid, score
    userid_list = []
    itemid_list = []
    for uid, iid_list in zip(t_data[0], t_data[1]):
        iid_list_ = iid_list.split(',')
        assert len(iid_list_) == 100
        userid_list += [uid] * 100
        itemid_list += iid_list_

    data = pd.DataFrame({
        # userId itemId	score
        "userId": userid_list,
        "itemId": itemid_list,
        "score" : [0] * len(itemid_list)
    })
    print('transformed data shape:', data.shape)
    return data

def get_ranking_raw_data(t_path):
    t_valid = get_valid_file(os.path.join(t_path, "valid_run.tsv"), os.path.join(t_path, "valid_qrel.tsv"))
    t_test = get_test_file(os.path.join(t_path, "test_run.tsv"))
    return t_valid, t_test

if __name__ == "__main__":
    s1_path = '../input/s1/'
    s2_path = '../input/s2/'
    s3_path = '../input/s3/'

    t1_path = '../input/t1/'
    t2_path = '../input/t2/'
    get_ranking_raw_data(t1_path)
    # get_gt_dict()
    

    