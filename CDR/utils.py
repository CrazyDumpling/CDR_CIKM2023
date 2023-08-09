# -*- coding: utf-8 -*-
import numpy as np
import torch
from collections import defaultdict

# environmental setting
NUM_USER = 10
NUM_ITEM = 1000


"Build several tools for simulation."
class McDropout(torch.nn.Module):
    def __init__(self, pdrop = 0.5):
        super(McDropout, self).__init__()
        assert 0 <= pdrop <= 1
        self.pdrop = pdrop

    def forward(self, X):
        mask = (torch.rand(X.shape) > self.pdrop).float()
        return mask * X


class LoggingSystem:
    def __init__(self, num_user = NUM_USER, num_item = NUM_ITEM):
        self.num_user = num_user
        self.num_item = num_item
        self.items_ = np.zeros(num_item)
        self.sample_ = None

    def update(self, logging):
        self.items_ += np.bincount(logging[:,1].astype(int), minlength=self.num_item)
        if self.sample_ is None:
            self.sample_ = logging
        else:
            self.sample_ = np.concatenate([self.sample_, logging], axis=0)

    @property
    def user_wise_ctr(self):
        ctr_list = []
        for i in range(self.num_user):
            user_sample = self.sample_[self.sample_[:,0] == i]
            ctr_list.append(user_sample[:, -1].mean())
        return np.array(ctr_list)

    @property
    def user_wise_ctr_last(self):
        # last 100 loggings ctr
        ctr_list = []
        for i in range(self.num_user):
            user_sample = self.sample_[self.sample_[:,0] == i]
            ctr_list.append(user_sample[-10:, -1].mean())
        return np.array(ctr_list)

    @property
    def item_wise_ctr(self):
        ctr_list = []
        for i in range(self.num_item):
            item_sample = self.sample_[self.sample_[:,1] == i]
            ctr_list.append(item_sample[:, -1].mean().astype(np.float32))
        return np.array(ctr_list)

    @property
    def item_stat(self):
        return self.items_.astype(int)

    @property
    def logs(self):
        return self.sample_

    @property    
    def logs_last(self):
        return self.sample_[-100:]

class Policy:
    """A policy to assign items to each user.
    """
    def __init__(self, model, num_user, num_item):
        # initialize a random policy
        self.policy = None
        # initialize the model
        self.model = model
        self.num_user = num_user
        self.num_item = num_item

    def learn(self, x, y, *args, **kwargs):
        # learn the policy
        if self.policy is None:
            """None indicates random policy.
            """
            self.policy = self.model

            if self.policy is not None:
                if kwargs["lamb"] is not None:
                    self.policy.fit(x,y, lamb = kwargs["lamb"])
                else:
                    # LR method
                    self.policy.fit(x,y)

        else:
            """Given MF or MF-CRRM model.
            """
            if kwargs["lamb"] is not None:
                self.policy.fit(x, y, lamb = kwargs["lamb"])
            else:
                self.policy.fit(x, y)

            return

    def predict(self, x_test):
        # predict the ctr rate
        if self.policy is None:
            return np.ones(x_test.shape[0]) * 0.5
        else:
            return self.policy.predict_proba(x_test)[:,1]

    def forward(self, user_idx, top_n=3):
        """Give user idx, gives recommended items.
        """
        if self.policy is None:
            # give random assignment
            pred = np.random.rand(len(user_idx), self.num_item)
        else:
            item_idx = np.arange(self.num_item)
            sample = []
            # generate test samples
            for user in user_idx:
                sample.extend([[user, item] for item in item_idx])

            x_test = np.array(sample)

            pred = self.policy.predict_proba(x_test)[:,1]
            # rank item
            pred = pred.reshape(self.num_user, self.num_item)

        # deterministically rank from top to end
        pred_item = np.argsort(-pred, 1)[:,:top_n]

        return pred_item

def simulate_click_or_not(pred_item, USER_PREF_MAT):
    """Given the predicted item index,
    simulate the user's click or not based on
    the given USER_PREF_MAT.
    """
    sample = []

    for user in range(USER_PREF_MAT.shape[0]):
        real_pref = USER_PREF_MAT[user][pred_item[user]]
        response = np.zeros_like(real_pref)
        # click
        response[np.random.rand(real_pref.shape[0]) < real_pref] = 1
        response_b = response.astype(bool)

        sample += [[user, item, label] for item,label in zip(pred_item[user],response)]

    sample_ar = np.array(sample)
    return sample_ar[:,:-1], sample_ar[:, -1]

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def gini_index(user_utility):
    from sklearn.metrics import auc
    cum_L = np.cumsum(np.sort(user_utility))
    sum_L = np.sum(user_utility)
    num_user = len(user_utility)
    xx = np.linspace(0,1,num_user + 1)
    yy = np.append([0],cum_L / sum_L)

    gi = (0.5 - auc(xx,yy)) / 0.5
    gu = sum_L / num_user

    print("Num User:", num_user)
    print("Gini index:", gi)
    print("Global utility:", gu)
    return gi, gu

def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y

def binarize(y, thres=3):
    """Given threshold, binarize the ratings.
    """
    y[y< thres] = 0
    y[y>=thres] = 1
    return y

def shuffle(x, y):
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]


def get_user_wise_ctr(x_test,y_test,test_pred,top_N=5):
    offset = 0
    user_idxs = np.unique(x_test[:,0])
    user_ctr_list = []
    for user in user_idxs:
        mask = x_test[:,0] == user
        pred_item = np.argsort(-test_pred[mask])[:top_N] + offset
        u_ctr = y_test[pred_item].sum() / pred_item.shape[0]
        user_ctr_list.append(u_ctr)
        offset += mask.sum()

    user_ctr_list = np.array(user_ctr_list)
    return user_ctr_list


def minU(x_test, y_test, test_pred, top_N=5):
    offset = 0
    user_idxs = np.unique(x_test[:,0])
    user_ctr_list = []
    for user in user_idxs:
        mask = x_test[:,0] == user
        pred_item = np.argsort(-test_pred[mask])[:top_N] + offset
        u_ctr = y_test[pred_item].sum() / pred_item.shape[0]
        user_ctr_list.append(u_ctr)
        offset += mask.sum()

    user_ctr_list = np.array(user_ctr_list)
    print("minU: {}, # of minU: {}".format(min(user_ctr_list), sum(user_ctr_list == min(user_ctr_list))))
    return user_ctr_list

def generate_rcts(num_sample, USER_PREF_MAT):
    num_user = USER_PREF_MAT.shape[0]
    num_item = USER_PREF_MAT.shape[1]

    idx1 = np.random.randint(0,num_user,num_sample)
    idx2 = np.random.randint(0,num_item,num_sample)

    pred = np.random.rand(USER_PREF_MAT.shape[0], USER_PREF_MAT.shape[1])
    pred_item = np.argsort(-pred, 1)[:,:3]

    mask = pred[idx1,idx2] > USER_PREF_MAT[idx1,idx2]
    y = mask.astype(int)
    x = np.concatenate([idx1.reshape(-1,1),
        idx2.reshape(-1,1)], axis=1)

    return x , y

def ndcg_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(top_k)].append(ndcg_k)

    return result_map

def recall_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate RECALL@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            hits = y_u[pred_top_k]

            all_hits = y_u[np.argsort(-y_u)][:top_k]

            if np.sum(all_hits) == 0:
                recall_k = 1
            else:
                recall_k = np.sum(hits) / np.sum(all_hits)

            result_map["recall_{}".format(top_k)].append(recall_k)

    return result_map

def mse_func(vector_predict,vector_true):
    mse = torch.mean((vector_predict - vector_true)**2)
    return mse.item()

def auc_func(vector_predict,vector_true):
    pos_indexes = torch.where(vector_true == 1)[0]
    pos_whe=(vector_true == 1)
    sort_indexes = torch.argsort(vector_predict)
    rank=torch.zeros((len(vector_predict)))
    rank[sort_indexes] = torch.FloatTensor(list(range(len(vector_predict))))
    rank = rank * pos_whe
    auc = (torch.sum(rank) - len(pos_indexes) * (len(pos_indexes) - 1) / 2)/ \
            (len(pos_indexes) * (len(vector_predict) - len(pos_indexes)))

    return auc.item()

#metric evaluate implement using python
def evaluate_func(model, dataset, top_k_list = [5,10]):
    num_user,num_item = dataset.get_user_item()
    binary_data = dataset.binary_data
    data = dataset.data
    result_map = defaultdict(list)
    result = defaultdict()
    for user in range(1,num_user):
        bdata = binary_data[binary_data[:,0] == user]
        if np.sum(bdata[:,2] == 1) == 0:
            continue

        pred = model.predict(torch.from_numpy(bdata[:,0:2]).to(torch.int64))

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred)[:top_k]
            log2_iplus1 = np.log2(1+np.arange(1,len(pred_top_k)+1))
            
            dcg = (bdata[pred_top_k,2] == 1) / log2_iplus1
            idcg = (bdata[np.argsort(-bdata[:,2]),2] == 1)[:top_k] / log2_iplus1
            ndcg = np.sum(dcg)/np.sum(idcg)

            hit = np.sum(bdata[pred_top_k,2] == 1)
            prec = hit/top_k

            all_hit = np.sum(bdata[:,2] == 1)
            recall = hit/all_hit
              
            result_map["ndcg_{}".format(top_k)].append(ndcg)
            result_map["prec_{}".format(top_k)].append(prec) 
            result_map["recall_{}".format(top_k)].append(recall)
        
    for top_k in top_k_list:
        result["ndcg_{}".format(top_k)] = np.mean(result_map["ndcg_{}".format(top_k)])
        result["prec_{}".format(top_k)] = np.mean(result_map["prec_{}".format(top_k)])
        result["recall_{}".format(top_k)] = np.mean(result_map["recall_{}".format(top_k)])

    vector_true = torch.from_numpy(binary_data[:,2]).detach()
    vector_lb = torch.from_numpy(data[:,2]).detach()
    vector_predict = model.predict(torch.from_numpy(data[:,0:2]).to(torch.int64))
    vector_predict = torch.Tensor(vector_predict)
    result["auc"] = auc_func(vector_predict,vector_true)
    result["mse"] = mse_func(vector_predict,vector_lb)

    return result

