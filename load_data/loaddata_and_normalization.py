#!python
# coding: utf-8

import numpy as np
from reader import read_svmlight
import scipy.io as sio

# # loaddata and normalization

def get_data_path():
    dataset_txt = 'example_data'
    dataset_mat = 'example_data/example_data'
    return dataset_txt, dataset_mat

def query_level_norm(X, QueryIds):
    X_norm = []
    for l in QueryIds:
        X_q1 = X[l]
        X_q1 = X_q1.todense()
        denominator = X_q1.max(axis=0)-X_q1.min(axis=0)
        denominator[denominator==0] = 1
        X_q1_norm = (X_q1 - X_q1.min(axis=0))/denominator
        X_norm += X_q1_norm.tolist()
    X_norm = np.array(X_norm)
    return X_norm

def load_data(filename):
    X, Y, QueryIds = read_svmlight(filename)
    X_norm = query_level_norm(X, QueryIds)
    for i, qs in enumerate(QueryIds):
        for j, q in enumerate(qs):
            QueryIds[i][j] += 1
    data = {'X':X_norm, 'Y':Y, 'qids':QueryIds}
    return data
    

def load_and_norm():
    dataset_txt_path, dataset_mat_path = get_data_path()
    print dataset_txt_path
    print dataset_mat_path

    f_load = '%s/' % (dataset_txt_path)
    data_train = load_data(f_load + 'train.txt')
    data_vali = load_data(f_load + 'vali.txt')
    data_test = load_data(f_load + 'test.txt')
    data = {
        'X_train':data_train['X'],
        'Y_train':data_train['Y'],
        'Q_train':data_train['qids'],
        'X_vali':data_vali['X'],
        'Y_vali':data_vali['Y'],
        'Q_vali':data_vali['qids'],
        'X_test':data_test['X'],
        'Y_test':data_test['Y'],
        'Q_test':data_test['qids']}
    f_store = '%s' % (dataset_mat_path)
    print '    ', f_load + '(train.txt vali.txt test.txt)'
    print ' -> ', f_store
    sio.savemat(f_store, data)
    print ''

load_and_norm()

