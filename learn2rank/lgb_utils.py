from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
import lightgbm as lgb
import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool, MetricVisualizer
from copy import deepcopy
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
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

def train_kFold_lgb_cls(lgb_valid_data, lgb_test_data, featureslist, cross_domain=False, groupname='userId', label='score', Kfold=10):
    '''
    train lgb with lambda rank loss.
    return valid_out, test_out
    '''

    lgb_valid_data['itemId'] = lgb_valid_data['itemId'].astype('category')
    lgb_test_data['itemId'] = lgb_test_data['itemId'].astype('category')
    zero_features, feature_importances = identify_zero_importance_features(lgb_valid_data[featureslist], lgb_valid_data['score'], 2)
    print(zero_features)
    print(feature_importances)
    featureslist = [fea for fea in featureslist if fea not in zero_features]
    
    print(featureslist)
    params = {
        'boosting_type': 'gbdt',
        'objective' : 'binary',# rank_xendcg, lambdarank # 756
        # 'objective' : 'binary',
        'metric': ['auc'],
        'num_leaves':9,
        'lambda_l1': 0.3,
        'lambda_l2': 0.5,
        'learning_rate': 0.037,
        'min_child_samples':20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'random_state':42,
        'num_threads':20
    }
    kFOLD_NUM = Kfold
    group_kfold = StratifiedKFold(n_splits=kFOLD_NUM, random_state=42, shuffle=True)   
    X, y = lgb_valid_data[featureslist], lgb_valid_data[label]
    test = lgb_test_data
    train_temp = lgb_valid_data[['userId', 'itemId']]

    oof_prob = np.zeros(len(X))
    preds = np.zeros(len(test)) 

    for train_index, test_index in group_kfold.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]


        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        lgb_model = lgb.train(
                params,
                dtrain,
                num_boost_round=1000000,
                valid_sets=[dvalid],
                early_stopping_rounds=50,
                verbose_eval=50,
                categorical_feature=['itemId'] if 'itemId' in featureslist else[]#categorical_feature,
            )
        
        feature_columns = featureslist 
        imp = get_lgbm_varimp(lgb_model, feature_columns, 100000000)
        oof_prob[test_index] = lgb_model.predict(X_valid)
        preds += lgb_model.predict(test[featureslist]) / Kfold
        print(imp.head(100))
        # print(imp.tail(30)) 
    from sklearn.metrics import roc_auc_score
    print('from sklearn.metrics import roc_auc_score: ', roc_auc_score(y, oof_prob))
    if not cross_domain:
        submit_columns = ['userId', 'itemId']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test
    else:
        submit_columns = ['userId', 'itemId', 'm']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test

def fit_model(default_parameters, loss_function, additional_params, train_pool, test_pool):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function
    
    if additional_params is not None:
        parameters.update(additional_params)
        
    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    
    return model

def train_kFold_catboost(lgb_valid_data, lgb_test_data, featureslist, cross_domain=False, groupname='userId', label='score', Kfold=5):
    '''
    train lgb with lambda rank loss.
    return valid_out, test_out
    '''

    lgb_valid_data['itemId'] = lgb_valid_data['itemId']#.astype('category')
    lgb_test_data['itemId'] = lgb_test_data['itemId']#.astype('category')
    
    print(featureslist)
    default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG:top=10'],
    'verbose': False,
    'random_seed': 0,
    'task_type': 'GPU'
    }

    kFOLD_NUM = Kfold
    group_kfold = StratifiedKFold(n_splits=kFOLD_NUM, random_state=42, shuffle=True)   
    X, y = lgb_valid_data[featureslist], lgb_valid_data[label]
    test = lgb_test_data

    oof_prob = np.zeros(len(X))
    preds = np.zeros(len(test)) 
    train_temp = lgb_valid_data[['userId', 'itemId']]

    for train_index, test_index in group_kfold.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        queries_train = train_temp.iloc[train_index]['userId']
        queries_test = train_temp.iloc[test_index]['userId']
        train_pool = Pool(
            data=X_train,
            label=y_train,
            group_id=queries_train
        )
        valid_pool = Pool(
            data=X_valid,
            label=y_valid,
            group_id=queries_test
        )

        test_pool = Pool(
            data=lgb_test_data[featureslist],
            label=lgb_test_data[label],
            group_id=lgb_test_data['userId']
        )
        model = fit_model(default_parameters, "YetiRankPairwise", None, train_pool, valid_pool)
        oof_prob[test_index] = model.predict(valid_pool)
        preds += model.predict(test_pool) / Kfold
        # print(imp.tail(30)) 

    from sklearn.metrics import roc_auc_score
    print('from sklearn.metrics import roc_auc_score: ', roc_auc_score(y, oof_prob))
    if not cross_domain:
        submit_columns = ['userId', 'itemId']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test
    else:
        submit_columns = ['userId', 'itemId', 'm']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test

def identify_zero_importance_features(train, train_labels, iterations=5):
    """
    Identify zero importance features in a training dataset based on the 
    feature importances from a gradient boosting model. 
    
    Parameters
    --------
    train : dataframe
        Training features
        
    train_labels : np.array
        Labels for training data
        
    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])
    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')
    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):
        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size = 0.25, random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc', verbose = 200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations
    
    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)
    
    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] <= 2.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))
    
    return zero_features, feature_importances

def train_kFold_lgb_ranking(lgb_valid_data, lgb_test_data, featureslist, cross_domain=False, groupname='userId', label='score', Kfold=10):
    '''
    train lgb with lambda rank loss.
    return valid_out, test_out
    '''
    zero_features, feature_importances = identify_zero_importance_features(lgb_valid_data[featureslist], lgb_valid_data['score'], 2)
    print(zero_features)
    print(feature_importances)
    featureslist = [fea for fea in featureslist if fea not in zero_features]

    lgb_valid_data['itemId'] = lgb_valid_data['itemId'].astype('category')
    lgb_test_data['itemId'] = lgb_test_data['itemId'].astype('category')
    
    # print(featureslist)
    params = {
        'boosting_type': 'gbdt',
        'objective' : 'lambdarank',# rank_xendcg, lambdarank # 756
        # 'objective' : 'binary',
        'metric': ['ndcg','auc'],
        "eval_at":[10],
        'num_leaves':24,
        'lambda_l1': 0.1,
        'lambda_l2': 0.5,
        # 'max_depth': -1,
        'learning_rate': 0.037,
        # 'min_child_samples':10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'random_state':42,
        'num_threads':20
    }
    kFOLD_NUM = Kfold
    group_kfold = GroupKFold(n_splits=kFOLD_NUM)   
    groups = lgb_valid_data[groupname]
    X, y = lgb_valid_data[featureslist], lgb_valid_data[label]
    print(featureslist)
    test = lgb_test_data
    train_temp = lgb_valid_data[['userId', 'itemId']]

    oof_prob = np.zeros(len(X))
    preds = np.zeros(len(test)) 
    print('--8'*100)
    for train_index, test_index in group_kfold.split(X, y, groups):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        # print(X_train.columns)

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
                num_boost_round=1000000,
                valid_sets=[dvalid],
                early_stopping_rounds=100,
                verbose_eval=50,
                categorical_feature=['itemId'] if 'itemId' in featureslist else[]#categorical_feature,
            )
        
        feature_columns = featureslist 
        print()
        imp = get_lgbm_varimp(lgb_model, featureslist, 100000000)
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
        submit_columns = ['userId', 'itemId', 'm']
        out_valid = lgb_valid_data[submit_columns]
        out_test = lgb_test_data[submit_columns]
        out_valid['score'] = oof_prob
        out_test['score'] = preds
        return out_valid, out_test
