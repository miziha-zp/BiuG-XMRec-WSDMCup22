###  df['{}-itemId_{}_mean'.format(col, col)] = df['itemId_{}_mean'.format(col)]-df[col]
not add 
===================== Market : t1t2=====================
======= Set val : score(ndcg10_val)=0.655034878170 =======
======= Set val : score(r10_val)=0.766597383543 =======
add 
===================== Market : t1t2=====================
======= Set val : score(ndcg10_val)=0.654085047659 =======
======= Set val : score(r10_val)=0.765007947182 =======
### 

nohup python lgb_ranking.py > logs/item2vec.log 2>&1 &



# 
1. -get_feature_here

# rank
0.6695
# 200 no earlystop
num_boost_round=200,
valid_sets=[dtrain, dvalid],
early_stopping_rounds=200,
verbose_eval=50,



# offline 663 add 600+features
0.6736
# add itemKNNsum(1,2,3) 6632
0.6745
# add sum_mf_lgcn_ultraGCN 6633
0.6740
# add sum_lgcn_ultraGCN 6633
0.6741
# add itemKNNsum(2,3) 
0.6726
# 上面的+xm
0.6740
# return itemKNN(1,2,3)
0.6745

remove _t1, _t2 ---> xm 6736

add EASE

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] -> [1, 2, 9, 10]