# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pre_process import preprocess_gender,preprocess_total_fee,preprocess_age
from model import build_model

import warnings
warnings.filterwarnings('ignore')

#============local data==============
train_path = 'dataset/train_all.csv'
test_path = 'dataset/republish_test.csv'

#============server data===============
# train_path = 'train_all.csv'
# test_path = 'republish_test.csv'
train = pd.read_csv(train_path,low_memory= False)
test= pd.read_csv(test_path,low_memory=False)

label2current_service = dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))
train['current_service'] = train['current_service'].map(current_service2label)
train_length = len(train)

test_length = len(test)
test_user_id = test['user_id']
test['service_type'].replace(3,4,inplace=True)
test['current_service'] = -1

#============combination with train and test==============
combined_train_test = train.append(test)
combined_train_test.reset_index(drop = True,inplace=True)
combined_train_test.drop(['user_id'],axis =1,inplace= True)

#=============features of package=============
def standard_total_fee(train):
    total_fee_of_service = train[['current_service', '1_total_fee']].groupby(['current_service'])['1_total_fee'].value_counts()
    #standard fee of package
    service_fee_list = []
    for i in range(train['current_service'].nunique()):
        service_fee_list.append(float(total_fee_of_service[i].index[0]))
    grouped_1_total_fee_median = train[['current_service', '1_total_fee']].groupby(['current_service'])['1_total_fee'].median()
    grouped_1_total_fee_mean = train[['current_service', '1_total_fee']].groupby(['current_service'])['1_total_fee'].mean()
    return service_fee_list, grouped_1_total_fee_median, grouped_1_total_fee_mean

def standard_online_time(train):
    grouped_online_time_median = train[['current_service','online_time']].groupby(['current_service'])['online_time'].median()
    grouped_online_time_mean = train[['current_service','online_time']].groupby(['current_service'])['online_time'].mean()
    return grouped_online_time_median, grouped_online_time_mean

def standard_traffic(train):
    grouped_month_traffic_median = train[['current_service','month_traffic']].groupby(['current_service'])['month_traffic'].median()
    grouped_month_traffic_mean = train[['current_service','month_traffic']].groupby(['current_service'])['month_traffic'].mean()
    grouped_last_month_traffic_median = train[['current_service','last_month_traffic']].groupby(['current_service'])['last_month_traffic'].median()
    grouped_last_month_traffic_mean = train[['current_service','last_month_traffic']].groupby(['current_service'])['last_month_traffic'].mean()
    grouped_local_last_month_traffic_median = train[['current_service','local_trafffic_month']].groupby(['current_service'])['local_trafffic_month'].median()
    grouped_local_last_month_traffic_mean = train[['current_service','local_trafffic_month']].groupby(['current_service'])['local_trafffic_month'].mean()
    return grouped_month_traffic_median,grouped_month_traffic_mean,grouped_last_month_traffic_median,grouped_last_month_traffic_mean,grouped_local_last_month_traffic_median,grouped_local_last_month_traffic_mean

def standard_caller(train):
    grouped_local_caller_time_median = train[['current_service','local_caller_time']].groupby(['current_service'])['local_caller_time'].median()
    grouped_local_caller_time_mean = train[['current_service','local_caller_time']].groupby(['current_service'])['local_caller_time'].mean()
    grouped_service1_caller_time_median = train[['current_service','service1_caller_time']].groupby(['current_service'])['service1_caller_time'].median()
    grouped_service1_caller_time_mean = train[['current_service','service1_caller_time']].groupby(['current_service'])['service1_caller_time'].mean()
    grouped_service2_caller_time_median = train[['current_service','service2_caller_time']].groupby(['current_service'])['service2_caller_time'].median()
    grouped_service2_caller_time_mean = train[['current_service','service2_caller_time']].groupby(['current_service'])['service2_caller_time'].mean()
    return grouped_local_caller_time_median,grouped_local_caller_time_mean,grouped_service1_caller_time_median,grouped_service1_caller_time_mean,grouped_service2_caller_time_median,grouped_service2_caller_time_mean

service_fee_list, grouped_1_total_fee_median, grouped_1_total_fee_mean = standard_total_fee(train)
grouped_online_time_median, grouped_online_time_mean = standard_online_time(train)
grouped_month_traffic_median,grouped_month_traffic_mean,grouped_last_month_traffic_median,grouped_last_month_traffic_mean,grouped_local_last_month_traffic_median,grouped_local_last_month_traffic_mean = standard_traffic(train)
grouped_local_caller_time_median,grouped_local_caller_time_mean,grouped_service1_caller_time_median,grouped_service1_caller_time_mean,grouped_service2_caller_time_median,grouped_service2_caller_time_mean = standard_caller(train)

#=========prepare for data=========
combined_train_test = preprocess_gender(combined_train_test)
combined_train_test = preprocess_total_fee(combined_train_test)
combined_train_test = preprocess_age(combined_train_test)
#=========feature enigeer===========

#===========online_time============
def process_online_time(data):
    # data['online_time_nothalfyear'] = data['online_time'].apply(lambda x: 1 if x <= 6 else 0)
    # data['online_time_notyear'] = data['online_time'].apply(lambda x: 1 if x <= 12 else 0)
    # data['online_time_notpastyear'] = data['online_time'].apply(lambda x: 1 if x <= 18 else 0)
    # data['online_time_nottwoyear'] = data['online_time'].apply(lambda x: 1 if x <= 24 else 0)
    i = 0
    for item in grouped_online_time_median:
        data['this_vs' + str(i) + '_online_time_median'] = data['online_time'] - item
        i += 1
    i = 0
    for item in grouped_online_time_mean:
        data['this_vs' + str(i) + '_online_time_mean'] = data['online_time'] - item
        i += 1
    return data

def process_total_fee(data):
    # i = 0
    # for item in service_fee_list:
    #     data['this_vs'+str(i) + '_standard_fee'] = data['1_total_fee'] - item
    #     i += 1
    i = 0
    for item in grouped_1_total_fee_median:
        data['this_vs' + str(i) + '_1total_fee_median'] = data['1_total_fee'] - item
        i += 1
    i = 0
    for item in grouped_1_total_fee_mean:
        data['this_vs' + str(i) + '_1total_fee_mean'] = data['1_total_fee'] - item
        i += 1

    def two_total_fee_fun(x):
        return (x['1_total_fee'] + (x['3_total_fee'] if x['3_total_fee'] >= 0 else 0) + (x['4_total_fee'] if x['4_total_fee'] >= 0 else 0)) / 3.0
    data.loc[data['2_total_fee'] < 0, '2_total_fee']  = data.loc[data['2_total_fee'] < 0].apply(two_total_fee_fun, axis=1)

    def three_total_fee_fun(x):
        return (x['1_total_fee'] + (x['2_total_fee'] if x['2_total_fee'] >= 0 else 0) + (x['4_total_fee'] if x['4_total_fee'] >= 0 else 0)) / 3.0
    data.loc[data['3_total_fee'] < 0, '3_total_fee'] = data.loc[data['3_total_fee'] < 0].apply(three_total_fee_fun, axis=1)

    def four_total_fee_fun(x):
        return (x['1_total_fee'] + (x['2_total_fee'] if x['2_total_fee'] >= 0 else 0) + (x['3_total_fee'] if x['3_total_fee'] >= 0 else 0)) / 3.0
    data.loc[data['4_total_fee'] < 0, '4_total_fee'] = data.loc[data['4_total_fee'] < 0].apply(four_total_fee_fun, axis=1)

    data['aver_total_fee'] = (data['1_total_fee'] + data['2_total_fee'] + data['3_total_fee'] + data['4_total_fee']) / 4.0
    return data

def process_traffic(data):
    i = 0
    for item in grouped_month_traffic_median:
        data['this_vs' + str(i) + '_month_traffic_median'] = data['month_traffic'] - item
        i += 1
    i = 0
    for item in grouped_month_traffic_mean:
        data['this_vs' + str(i) + '_month_traffic_mean'] = data['month_traffic'] - item
        i += 1

    i = 0
    for item in grouped_last_month_traffic_median:
        data['this_vs' + str(i) + '_last_month_traffic_median'] = data['last_month_traffic'] - item
        i += 1
    i = 0
    for item in grouped_last_month_traffic_mean:
        data['this_vs' + str(i) + '_last_month_traffic_mean'] = data['last_month_traffic'] - item
        i += 1

    i = 0
    for item in grouped_local_last_month_traffic_median:
        data['this_vs' + str(i) + '_local_last_month_traffic_median'] = data['local_trafffic_month'] - item
        i += 1
    i = 0
    for item in grouped_local_last_month_traffic_mean:
        data['this_vs' + str(i) + '_local_last_month_traffic_mean'] = data['local_trafffic_month'] - item
        i += 1

    def traffic_fun(data):
        if data['month_traffic'] >= data['local_trafffic_month']:
            return 1
        else:
            return 0
    data['month_upper_local'] = data.apply(traffic_fun, axis=1)

    def is_have_last_traffic(data):
        if data['last_month_traffic'] != 0:
            return 1
        else:
            return 0
    data['is_have_last_traffic'] = data.apply(is_have_last_traffic, axis=1)

    return data



def process_caller(data):
    i = 0
    for item in grouped_local_caller_time_median:
        data['this_vs' + str(i) + '_local_caller_time_median'] = data['local_caller_time'] - item
        i += 1
    i = 0
    for item in grouped_local_caller_time_mean:
        data['this_vs' + str(i) + '_local_caller_time_mean'] = data['local_caller_time'] - item
        i += 1

    i = 0
    for item in grouped_service1_caller_time_median:
        data['this_vs' + str(i) + '_service1_caller_time_median'] = data['service1_caller_time'] - item
        i += 1
    i = 0
    for item in grouped_service1_caller_time_mean:
        data['this_vs' + str(i) + '_service1_caller_time_mean'] = data['service1_caller_time'] - item
        i += 1

    i = 0
    for item in grouped_service2_caller_time_median:
        data['this_vs' + str(i) + '_service2_caller_time_median'] = data['service2_caller_time'] - item
        i += 1
    i = 0
    for item in grouped_service2_caller_time_mean:
        data['this_vs' + str(i) + '_service2_caller_time_mean'] = data['service2_caller_time'] - item
        i += 1

    data['caller_time'] = data['local_caller_time'] + data['service1_caller_time'] + data['service2_caller_time']

    def local_upper_ser1(data):
        if data['local_caller_time'] >= data['service1_caller_time']:
            return 1
        else:
            return 0
    data['local_upper_ser1'] = data.apply(local_upper_ser1, axis=1)
    return data

def process_pay(data):
    data['aver_pay_num'] = data['pay_num'] / data['pay_times']
    return data

#-------  many_over_bill--------
#---------contract_type----------
# contract_type_dummy = pd.get_dummies(combined_train_test['contract_type'],prefix = 'contract_type')
# combined_train_test = pd.concat([combined_train_test,contract_type_dummy],axis  = 1)
# combined_train_test.drop(['contract_type'],axis = 1,inplace = True)
#--------contract_time---------
#-------is_promise_low_consume	--------
#-------net_service-------
# net_service_dummy = pd.get_dummies(combined_train_test['net_service'],prefix='net_service')
# combined_train_test = pd.concat([combined_train_test,net_service_dummy],axis = 1)
# combined_train_test.drop(['net_service'],axis = 1,inplace = True)
#-------pay_times and  pay_num-----------
#-------gender---------
# gender_dummy = pd.get_dummies(combined_train_test['gender'],prefix= 'gender')
# combined_train_test = pd.concat([combined_train_test,gender_dummy],axis = 1)
# combined_train_test.drop(['gender'],axis = 1,inplace = True)
#-------age--------
# def age_fun(val):
#     if val == 0:
#         return 0
#     if val > 0 and val < 15:
#         return 1
#     elif val >= 15 and val < 25:
#         return 2
#     elif val >= 25 and val < 60:
#         return 3
#     elif val >= 60 and val < 90:
#         return 4
#     else:
#         return 5
# combined_train_test['age_range'] = combined_train_test['age'].apply(age_fun)
# age_dummy = pd.get_dummies(combined_train_test['age_range'],prefix= 'age_range')
# combined_train_test = pd.concat([combined_train_test,age_dummy],axis = 1)
# combined_train_test.drop(['age_range','age'],axis = 1,inplace = True)
#---------complaint_level---------
# complaint_level_dummy = pd.get_dummies(combined_train_test['complaint_level'],prefix= 'complaint_level')
# combined_train_test = pd.concat([combined_train_test,complaint_level_dummy],axis = 1)
# combined_train_test.drop(['complaint_level'],axis = 1,inplace = True)
#----------former_complaint_fee---------
# combined_train_test.drop(['former_complaint_num'],axis = 1,inplace = True)
# combined_train_test.drop(['former_complaint_fee'],axis = 1,inplace = True)

combined_train_test = process_online_time(combined_train_test)
combined_train_test = process_total_fee(combined_train_test)
combined_train_test = process_traffic(combined_train_test)
combined_train_test = process_caller(combined_train_test)
combined_train_test = process_pay(combined_train_test)

#==========split===========
train = combined_train_test.loc[0:train_length-1]
test = combined_train_test.loc[train_length:]
test.drop(['current_service'],axis = 1,inplace = True)
Y = train.pop('current_service')
X = train

#===========tune================
build_model(X,Y,test,test_user_id,label2current_service)










