# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)

import dataset
from MF_mcdropout import MF_DR_MCDROPOUT, MF_MRDR_MCDROPOUT, MF_EIB_MCDROPOUT
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, evaluate_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

# load data
test_dataset = dataset.kuairand_dataset('test')
val_dataset = dataset.kuairand_dataset('val')
train_dataset = dataset.kuairand_dataset('train', ips = True, y_ips = val_dataset.get_sample(0.05))

num_user,num_item = train_dataset.get_user_item()
impute_dataset = dataset.kuairand_impute_dataset(num_user, num_item)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 2048, 
                                           shuffle = True)
impute_loader = torch.utils.data.DataLoader(dataset = impute_dataset,
                                            batch_size = 2048,
                                            shuffle = True)

"MF EIB MC"
mf_eib_mc = MF_EIB_MCDROPOUT(num_user, num_item)

mf_eib_mc.fit_with_dataloader(train_loader, impute_loader,
    lr=0.05,
    lamb=1e-4,
    tol=1e-5,
    verbose=True)

res = evaluate_func(mf_eib_mc, test_dataset)

print("***"*5 + "[MF-EIB-MC]" + "***"*5)
print("[MF-EIB-MC] test mse:", res["mse"])
print("[MF-EIB-MC] test auc:", res["auc"])
print("[MF-EIB-MC] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
        res["ndcg_5"], res["ndcg_10"]))
print("[MF-EIB-MC] recall@5:{:.6f}, recall@10:{:.6f}".format(
        res["recall_5"], res["recall_10"]))

print("***"*5 + "[MF-EIB-MC]" + "***"*5)

