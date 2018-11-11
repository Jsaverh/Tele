# -*- coding:utf-8 -*-
def preprocess_gender(data):
    def change(x):
        if x == 1 or x == '1' or x == '01':
            return 0
        elif x == 2 or x == '2' or x == '02':
            return 2
        else:
            return 0
    data['gender'].replace('\\N', 0,inplace = True)
    data['gender'] = data['gender'].apply(change)
    data['gender'] = data['gender'].astype(float)
    return data
def preprocess_age(data):
    data['age'].replace('\\N', 0, inplace=True)
    data['age'] = data['age'].astype(float)
    return data
def preprocess_total_fee(data):
    data['2_total_fee'].replace('\\N', -1, inplace=True)
    data['3_total_fee'].replace('\\N', -1, inplace=True)
    data['2_total_fee'] = data['2_total_fee'].astype(float)
    data['3_total_fee'] = data['3_total_fee'].astype(float)
    # combined_train_test['2_total_fee'].apply(lambda x:max(x,0))
    # combined_train_test['3_total_fee'].apply(lambda x:max(x,0))
    # combined_train_test['4_total_fee'].apply(lambda x:max(x,0))
    # combined_train_test.loc[combined_train_test['2_total_fee'] < 0,'2_total_fee'] = combined_train_test[combined_train_test['2_total_fee'] < 0]['1_total_fee']
    # combined_train_test.loc[combined_train_test['3_total_fee'] < 0,'3_total_fee'] = combined_train_test[combined_train_test['3_total_fee'] < 0]['1_total_fee']
    # combined_train_test.loc[combined_train_test['4_total_fee'] < 0,'4_total_fee'] = combined_train_test[combined_train_test['4_total_fee'] < 0]['1_total_fee']
    return data

