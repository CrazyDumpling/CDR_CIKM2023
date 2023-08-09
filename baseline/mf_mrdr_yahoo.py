# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb

from dataset import load_data
from MF import MF_MRDR
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, recall_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

dataset_name = "yahoo"

if dataset_name == "coat":
    train_mat, test_mat = load_data("coat")        
    x_train, y_train = rating_mat_to_sample(train_mat)
    x_test, y_test = rating_mat_to_sample(test_mat)
    num_user = train_mat.shape[0]
    num_item = train_mat.shape[1]

elif dataset_name == "yahoo":
    x_train, y_train, x_test, y_test = load_data("yahoo")
    x_train, y_train = shuffle(x_train, y_train)
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1

print("# user: {}, # item: {}".format(num_user, num_item))
# binarize
y_train = binarize(y_train)
y_test = binarize(y_test)

"MF MRDR MC"
mf_mrdr = MF_MRDR(num_user, num_item)

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

mf_mrdr.fit(x_train, y_train,  y_ips=y_ips,
    lr=0.05,
    batch_size=2048,
    lamb=1e-4,
    tol=1e-5,
    verbose=False)
test_pred = mf_mrdr.predict(x_test)
mse_mfmrdrmc = mse_func(y_test, test_pred)
auc_mfmrdrmc = roc_auc_score(y_test, test_pred)
ndcg_res = ndcg_func(mf_mrdr, x_test, y_test)
recall_res = recall_func(mf_mrdr, x_test, y_test)

print("***"*5 + "[MF-MRDR]" + "***"*5)
print("[MF-MRDR] test mse:", mse_mfmrdrmc)
print("[MF-MRDR] test auc:", auc_mfmrdrmc)
print("[MF-MRDR] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
        np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
print("[MF-MRDR] recall@5:{:.6f}, recall@10:{:.6f}".format(
        np.mean(recall_res["recall_5"]), np.mean(recall_res["recall_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[MF-MRDR]" + "***"*5)