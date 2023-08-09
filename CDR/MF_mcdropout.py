import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
import pdb
from utils import McDropout, gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def corrcoef(x,y):
    x = x-torch.mean(x)
    y = y-torch.mean(y)
    return ((x*y).mean()) / (torch.std(x) * torch.std(y))

def count_relate(model_pre,model_imp,x,y):
    y_pred = model_pre.sigmoid(model_pre.predict(x))
    y_imp = model_imp.sigmoid(model_imp.predict(x))
    loss_hat = F.binary_cross_entropy(y_imp,y,reduction='none')
    loss = F.binary_cross_entropy(y_pred,y,reduction='none')
    count_relate = torch.sum(loss < torch.abs(loss-loss_hat)).cpu().numpy()
    return count_relate


class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

        self.init_embedding()

    def init_embedding(self): 
        nn.init.xavier_normal_(self.W.weight)
        nn.init.xavier_normal_(self.H.weight)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class MF_McdropoutModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k = 4, pdrop = 0.5, *args, **kwargs):
        super(MF_McdropoutModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.dropout = McDropout(pdrop)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
        

    def sample_predict(self, x, Nsamples = 10):
        prediction = torch.zeros([Nsamples,x.shape[0]])
        for i in range(Nsamples):    
            user_idx = torch.LongTensor(x[:, 0])
            item_idx = torch.LongTensor(x[:, 1])
            U_emb = self.dropout(self.W(user_idx))
            V_emb = self.dropout(self.H(item_idx))

            prediction[i] = torch.sum(U_emb.mul(V_emb), 1).detach().cpu()
        
        prediction = self.dropout(self.sigmoid(prediction))

        return prediction
    
    def sample_bce(self, pred, imputation, Nsamples = 10):
        imputation_loss = torch.zeros([Nsamples, pred.shape[0]])
        for i in range(Nsamples):
            imputation_loss[i] = (F.binary_cross_entropy(pred, imputation[i], reduction="none"))
        
        res = torch.mean(imputation_loss, dim = 0)
        uncertainty = torch.mean((imputation_loss - res) ** 2, dim = 0)
        return res, uncertainty

    def predict(self, x):
        pred = self.predict_forward(x)
        return pred.detach().cpu()

    
class MF_DR_MCDROPOUT(nn.Module):
    def __init__(self, num_users, num_items, embedding_k = 4, un_thres = 10, pdrop = 0.5, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.un_thres = un_thres
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_McdropoutModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, pdrop = pdrop)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-5, G=1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]                

                sub_y = torch.Tensor(sub_y)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def fit_with_dataloader(self, train_loader, impute_loader, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, tol=1e-5,
        G = 1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)

        early_stop = 0
        for epoch in range(num_epoch):
            epoch_loss = 0

            for (sub_x, sub_y, inv_prop), (x_sampled) in zip(train_loader, impute_loader):
                # mini-batch training
                sub_x = sub_x.to(torch.int64)
                x_sampled = x_sampled.to(torch.int64)
                sub_y = sub_y.to(torch.float)            
                inv_prop = inv_prop.to(torch.float)   

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl    
    
    
class MF_MRDR_MCDROPOUT(nn.Module):
    def __init__(self, num_users, num_items, embedding_k = 4, un_thres = 10, pdrop = 0.5, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.un_thres = un_thres
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_McdropoutModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, pdrop = pdrop)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, x_test = None, y_test = None, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items) 

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x)
                pred = self.sigmoid(pred)

                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]] # batch size
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()      
                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss # batch size
                
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()

                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])                              
                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()


                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop ** 2 ) * (1 - 1 / inv_prop)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                      
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_MRDR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 5 == 0 and verbose:
                ndcg_res = ndcg_func(self, x_test, y_test)
                imputation_y_test = self.imputation.sample_predict(x_test)
                imputation_loss = F.binary_cross_entropy(imputation_y_test,torch.Tensor(y_test),reduction='none')
                count_related = count_relate(self.prediction_model,self.imputation,x_test,torch.Tensor(y_test))/len(x_test)

            if epoch == num_epoch - 1:
                print("[MF_MRDR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def fit_with_dataloader(self, train_loader, impute_loader, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, tol=1e-5,
        G = 1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)

        early_stop = 0
        for epoch in range(num_epoch):
            epoch_loss = 0

            for (sub_x, sub_y, inv_prop), (x_sampled) in zip(train_loader, impute_loader):
                # mini-batch training
                sub_x = sub_x.to(torch.int64)
                x_sampled = x_sampled.to(torch.int64)
                sub_y = sub_y.to(torch.float)            
                inv_prop = inv_prop.to(torch.float)   

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop ** 2 ) * (1 - 1 / inv_prop)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl
    
class MF_EIB_MCDROPOUT(nn.Module):
    def __init__(self, num_users, num_items, embedding_k = 4, un_thres = 10, pdrop = 0.5, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.un_thres = un_thres
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_McdropoutModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, pdrop = pdrop)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-5, G=1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]                

                sub_y = torch.Tensor(sub_y)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (e_hat_loss).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def fit_with_dataloader(self, train_loader, impute_loader, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, tol=1e-5,
        G = 1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)

        early_stop = 0
        for epoch in range(num_epoch):
            epoch_loss = 0

            for (sub_x, sub_y, inv_prop), (x_sampled) in zip(train_loader, impute_loader):
                # mini-batch training
                sub_x = sub_x.to(torch.int64)
                x_sampled = x_sampled.to(torch.int64)
                sub_y = sub_y.to(torch.float)            
                inv_prop = inv_prop.to(torch.float)   

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (e_hat_loss).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl    
    
class MF_DR_BIAS_MCDROPOUT(nn.Module):
    def __init__(self, num_users, num_items, embedding_k = 4, un_thres = 10, pdrop = 0.5, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.un_thres = un_thres
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_McdropoutModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, pdrop = pdrop)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-5, G=1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]                

                sub_y = torch.Tensor(sub_y)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop ** 3 ) * ((1 - 1 / inv_prop) ** 2)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def fit_with_dataloader(self, train_loader, impute_loader, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, tol=1e-5,
        G = 1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)

        early_stop = 0
        for epoch in range(num_epoch):
            epoch_loss = 0

            for (sub_x, sub_y, inv_prop), (x_sampled) in zip(train_loader, impute_loader):
                # mini-batch training
                sub_x = sub_x.to(torch.int64)
                x_sampled = x_sampled.to(torch.int64)
                sub_y = sub_y.to(torch.float)            
                inv_prop = inv_prop.to(torch.float)   

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                imputation_loss = (imputation_loss * (imputation_uncertainty < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                direct_loss = (direct_loss * (direct_uncertainty < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop ** 3 ) * ((1 - 1 / inv_prop) ** 2)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl    
    

class MF_DR_ADJ_MCDROPOUT(nn.Module):
    def __init__(self, num_users, num_items, embedding_k = 4, un_thres = 10, pdrop = 0.5, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.un_thres = un_thres
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_McdropoutModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, pdrop = pdrop)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0,  theta=1,
        tol=1e-5, G=1, verbose = False, Nsamples = 10): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9
        # G = int(2 * G / self.un_thres)
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]                

                sub_y = torch.Tensor(sub_y)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.sample_predict(sub_x, Nsamples)
                pred = self.sigmoid(pred)

                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.sample_predict(x_sampled, Nsamples)
                pred_u = self.sigmoid(pred_u)     
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="none") # o*eui/pui
                imputation_loss, imputation_uncertainty = self.imputation.sample_bce(pred, imputation_y)
                l1 = torch.abs(torch.mean(imputation_loss) - torch.mean(xent_loss)).detach()
                imputation_loss = (imputation_loss * ((l1 * theta + imputation_uncertainty) < self.un_thres)).sum()

                # print(f"imputation_y1.mean:{torch.mean(imputation_y1)}")
                # print(f"imputation_y.mean:{torch.mean(imputation_y)}")

                ips_loss = xent_loss.sum() - imputation_loss
                
                # direct loss
                direct_loss, direct_uncertainty = self.imputation.sample_bce(pred_u, imputation_y1)
                l2 = torch.abs(torch.mean(direct_loss) - torch.mean(xent_loss)).detach()
                # print(f"direct_loss:{direct_loss}")
                # print(f"xent_loss:{xent_loss}")
                # print(f"l1:{l1}")
                # print(f"l2:{l2}")

                direct_loss = (direct_loss * ((direct_uncertainty + l2 * theta) < self.un_thres)).sum()
                # print(f"uncertainty_y1:{uncertainty_y1}")
                # print(f"mean_uncertainty_y1:{torch.mean(uncertainty_y1)}")
                # print()
                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.sum().detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF_DR_MCDROPOUT] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF_DR_MCDROPOUT] Reach preset epochs, it seems does not converge.")

    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl