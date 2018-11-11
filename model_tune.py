from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
def lgb_tune(X,Y):
    print('开始调参')
    lgb_cf = lgb.LGBMClassifier(objective= 'muticlass')
    lgb_cf_param_grid = {
        # 'learning_rate':[0.05,0.1,0.15,0.2],
        # 'lambda_l1':[0,0.1,0.2,0.3],
        # "lambda_l2":[0.0,0.1,0.2,0.3],
        'num_leaves':[100,200,300],
        'max_depth':[5,6,7,8],
        # 'max_depth':[6,7,8],
        # 'feature_fraction':[0.0,0.2,0.4,0.6,0.8,1.0],
        # 'bagging_fraction':[0.0,0.2,0.4,0.6,0.8,1.0],
        'num_class':[15]
    }
    lgb_cf_grid = GridSearchCV(lgb_cf,lgb_cf_param_grid,cv = 3,scoring='f1_macro')
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=10)
    lgb_cf_grid.fit(X_train,Y_train)
    print('best para' + str(lgb_cf_grid.best_params_))
    print('best lgb score' + str(lgb_cf_grid.best_score_))
    print('train error' + str(lgb_cf_grid.score(X_test,Y_test)))