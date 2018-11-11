
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def build_model(X,Y,test,test_user_id,label2current_service):

    def f1_score_vali(preds, data_vali):
        labels = data_vali.get_label()
        preds = np.argmax(preds.reshape(11, -1), axis=0)
        score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
        return 'f1_score', score_vali ** 2, True

    def lgb_model(X,Y,test):
        lgb_score = []
        lgb_pred = []
        params = {
            'learning_rate': 0.03,
            # 'lambda_l1':0.1,
            'lambda_l2': 0.1,
            'num_leaves': 150,
            # 'max_bin':300,
            'objective': 'multiclass',
            'num_class': 11,
            'seed': 2018,
        }
        skf = StratifiedKFold(n_splits=5, random_state=40, shuffle=True)
        for index, (train_index, test_index) in enumerate(skf.split(X, Y)):
            print('lgb model 第' + str(index) + '次遍历：')
            X_train, X_valid, Y_train, Y_valid = X.loc[train_index], X.loc[test_index], Y.loc[train_index], Y.loc[test_index]
            train_data = lgb.Dataset(X_train, label=Y_train)
            validation_data = lgb.Dataset(X_valid, label=Y_valid)
            clf = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[validation_data],
                            early_stopping_rounds=300, feval=f1_score_vali, verbose_eval=1)

            xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
            xx_pred = [np.argmax(x) for x in xx_pred]

            xx_f1 = f1_score(Y_valid, xx_pred, average='macro')
            xx_f1 = xx_f1 * xx_f1
            lgb_score.append(xx_f1)

            y_test = clf.predict(test, num_iteration=clf.best_iteration)
            y_test = [np.argmax(x) for x in y_test]

            if index == 0:
                lgb_pred = np.array(y_test).reshape(-1, 1)
            else:
                lgb_pred = np.hstack((lgb_pred, np.array(y_test).reshape(-1, 1)))
        return lgb_score, lgb_pred
    def xgb_model(X,Y,test):
        xgb_score = []
        xgb_pred = []
        params = {
            'max_depth':12,
            'learning_rate':0.05,
            'objective': "multi:softprob",
            'subsample':1,
            'colsample_bytree':0.9,
            'colsample_bylevel':0.9,
            'reg_alpha':1,
            'reg_lambda':1,
            'scale_pos_weight':1,
            'base_score':0.5,
            'seed':2018,
            'num_class':11,
            'eval_metric':'mlogloss',

        }
        skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        for index, (train_index, test_index) in enumerate(skf.split(X, Y)):
            print('xgb model 第' + str(index) + '次遍历：')
            X_train, X_valid, Y_train, Y_valid = X.loc[train_index], X.loc[test_index], Y.loc[train_index], Y.loc[test_index]

            train_data = xgb.DMatrix(X_train, Y_train)
            validation_data = xgb.DMatrix(X_valid, Y_valid)
            watch_list = [(validation_data,'eval')]
            clf = xgb.train(params, train_data, 2000,watch_list,early_stopping_rounds=200)

            xx_pred = clf.predict(xgb.DMatrix(X_valid))
            xx_pred = [np.argmax(x) for x in xx_pred]

            xx_f1 = f1_score(Y_valid, xx_pred, average='macro')
            xx_f1 = xx_f1 * xx_f1
            xgb_score.append(xx_f1)

            y_test = clf.predict(xgb.DMatrix(test))
            y_test = [np.argmax(x) for x in y_test]

            if index == 0:
                xgb_pred = np.array(y_test).reshape(-1, 1)
            else:
                xgb_pred = np.hstack((xgb_pred, np.array(y_test).reshape(-1, 1)))
        return xgb_score,xgb_pred

    lgb_score,lgb_pred = lgb_model(X,Y,test)
    print('lgb结束')
    xgb_score,xgb_pred = xgb_model(X,Y,test)
    print('xgb结束')
    pred = np.stack((lgb_pred,xgb_pred))
    score = lgb_score.extend(xgb_score)

    submit = []
    for line in pred:
        submit.append(np.argmax(np.bincount(line)))
    df_test = pd.DataFrame()
    df_test['id'] = list(test_user_id.unique())
    df_test['predict'] = submit
    df_test['predict'] = df_test['predict'].map(label2current_service)
    df_test.to_csv('result.csv', index=False)
    print(score)