from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def get_lgbm_varimp(model, train_columns, max_vars=500000000):
    
    if "basic.Booster" in str(model.__class__):
        # lightgbm.basic.Booster was trained directly, so using feature_importance() function 
        cv_varimp_df = pd.DataFrame([train_columns, model.feature_importance()]).T
    else:
        # Scikit-learn API LGBMClassifier or LGBMRegressor was fitted, 
        # so using feature_importances_ property
        cv_varimp_df = pd.DataFrame([train_columns, model.feature_importances_]).T

    cv_varimp_df.columns = ['feature_name', 'varimp']

    cv_varimp_df.sort_values(by='varimp', ascending=False, inplace=True)

    cv_varimp_df = cv_varimp_df.iloc[0:max_vars]   

    return cv_varimp_df

    
def train_kFold_bce_ranking(lgb_valid_data, lgb_test_data, featureslist, cross_domain=False, groupname='userId', label='score', Kfold=5):
    '''
    train lgb with lambda rank loss.
    return valid_out, test_out
    '''

    lgb_valid_data['itemId'] = lgb_valid_data['itemId'].astype('category')
    lgb_test_data['itemId'] = lgb_test_data['itemId'].astype('category')
    
    print(featureslist)
    params = {
        'boosting_type': 'gbdt',
        'objective' : 'binary',
        'metric': ['ndcg'],
        "eval_at":[10],
        'num_leaves':32,
        'lambda_l1': 1,
        'lambda_l2': 1,
        # 'max_depth': -1,
        'learning_rate': 0.037,
        'min_child_samples':20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'random_state':42,
        'num_threads':-1
    }
    kFOLD_NUM = Kfold
    group_kfold = GroupKFold(n_splits=kFOLD_NUM)   
    groups = lgb_valid_data[groupname]
    X, y = lgb_valid_data[featureslist], lgb_valid_data[label]
    test = lgb_test_data
    train_temp = lgb_valid_data[['userId', 'itemId']]

    oof_prob = np.zeros(len(X))
    preds = np.zeros(len(test)) 

    for train_index, test_index in group_kfold.split(X, y, groups):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        
        fold_temp = train_temp.iloc[train_index]
        fold_temp['group'] = fold_temp.groupby('userId')['userId'].transform('size')
        temp = fold_temp[['userId','group']].drop_duplicates()
        train_group = temp['group'].tolist()
        print(sum(train_group))

        fold_temp = train_temp.iloc[test_index]
        fold_temp['group'] = fold_temp.groupby('userId')['userId'].transform('size')
        temp = fold_temp[['userId','group']].drop_duplicates()
        test_group = temp['group'].tolist()
        print(sum(test_group))

        dtrain = lgb.Dataset(X_train, label=y_train, group=train_group)
        dvalid = lgb.Dataset(X_valid, label=y_valid, group=test_group)

        lgb_model = lgb.train(
                params,
                dtrain,
                num_boost_round=10000,
                valid_sets=[dtrain, dvalid],
                early_stopping_rounds=50,
                verbose_eval=200,
                categorical_feature=['itemId'] if 'itemId' in featureslist else[]#categorical_feature,
            )
        
        feature_columns = featureslist 
        imp = get_lgbm_varimp(lgb_model, feature_columns, 100000000)
        oof_prob[test_index] = lgb_model.predict(X_valid)
        preds += lgb_model.predict(test[featureslist]) / Kfold
        print(imp.head(100000))
        # print(imp.tail(30)) 
    if not cross_domain:
        submit_columns = ['userId', 'itemId']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test
    else:
        submit_columns = ['userId', 'itemId','domain']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test

def train_kFold_lgb_ranking(lgb_valid_data, lgb_test_data, featureslist, cross_domain=False, groupname='userId', label='score', Kfold=5):
    '''
    train lgb with lambda rank loss.
    return valid_out, test_out
    '''

    lgb_valid_data['itemId'] = lgb_valid_data['itemId'].astype('category')
    lgb_test_data['itemId'] = lgb_test_data['itemId'].astype('category')
    
    print(featureslist)
    params = {
        'boosting_type': 'gbdt',
        'objective' : 'lambdarank',
        'metric': ['ndcg'],
        "eval_at":[10],
        'num_leaves':32,
        'lambda_l1': 1,
        'lambda_l2': 1,
        # 'max_depth': -1,
        'learning_rate': 0.037,
        'min_child_samples':20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'random_state':42,
        'num_threads':-1
    }
    kFOLD_NUM = Kfold
    group_kfold = GroupKFold(n_splits=kFOLD_NUM)   
    groups = lgb_valid_data[groupname]
    X, y = lgb_valid_data[featureslist], lgb_valid_data[label]
    test = lgb_test_data
    train_temp = lgb_valid_data[['userId', 'itemId']]

    oof_prob = np.zeros(len(X))
    preds = np.zeros(len(test)) 

    for train_index, test_index in group_kfold.split(X, y, groups):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        
        fold_temp = train_temp.iloc[train_index]
        fold_temp['group'] = fold_temp.groupby('userId')['userId'].transform('size')
        temp = fold_temp[['userId','group']].drop_duplicates()
        train_group = temp['group'].tolist()
        print(sum(train_group))

        fold_temp = train_temp.iloc[test_index]
        fold_temp['group'] = fold_temp.groupby('userId')['userId'].transform('size')
        temp = fold_temp[['userId','group']].drop_duplicates()
        test_group = temp['group'].tolist()
        print(sum(test_group))

        dtrain = lgb.Dataset(X_train, label=y_train, group=train_group)
        dvalid = lgb.Dataset(X_valid, label=y_valid, group=test_group)

        lgb_model = lgb.train(
                params,
                dtrain,
                num_boost_round=10000,
                valid_sets=[dtrain, dvalid],
                early_stopping_rounds=50,
                verbose_eval=200,
                categorical_feature=['itemId'] if 'itemId' in featureslist else[]#categorical_feature,
            )
        
        feature_columns = featureslist 
        imp = get_lgbm_varimp(lgb_model, feature_columns, 100000000)
        oof_prob[test_index] = lgb_model.predict(X_valid)
        preds += lgb_model.predict(test[featureslist]) / Kfold
        print(imp.head(100))
        # print(imp.tail(30)) 
    if not cross_domain:
        submit_columns = ['userId', 'itemId']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test
    else:
        submit_columns = ['userId', 'itemId','domain']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test