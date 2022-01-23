import os
from os.path import join as pjoin
target_path = 'ItemKNN/data/data'
for dataset  in ['t1', 't2', 't1_testing', 't2_testing']:
    t_target_path = pjoin(target_path, dataset)
    if not os.path.exists(t_target_path):
        print('create dir:', t_target_path)
        os.makedirs(t_target_path)
    f = open(pjoin(f'../../LightGCN-PyTorch-master/data/{dataset}', 'train.txt'))
    f2=open(pjoin(t_target_path, 'train_enmf.txt'),'w')
    f2.write('uid'+'\t'+'sid'+'\n')
    for line in f:
        str=line.strip().split()
        for j in range(1,len(str)):
            f2.write(str[0]+'\t'+str[j]+'\n')

    f = open(pjoin(f'../../LightGCN-PyTorch-master/data/{dataset}', 'test.txt'))
    f2=open(pjoin(t_target_path, 'test_enmf.txt'),'w')
    f2.write('uid'+'\t'+'sid'+'\n')
    for line in f:
        str=line.strip().split()
        for j in range(1,len(str)):
            f2.write(str[0]+'\t'+str[j]+'\n')