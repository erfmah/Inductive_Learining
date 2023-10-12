#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:58:09 2023

@author: pnaddaf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:51:23 2022

@author: pnaddaf
"""
import sys
import os
import argparse
import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import csv
from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv
import copy
from dataCenter import *
from utils import *
from models import *

from sklearn.preprocessing import OneHotEncoder
import pn2_helper as helper
import statistics
import warnings

warnings.simplefilter('ignore')

# %%  arg setup

##################################################################


parser = argparse.ArgumentParser(description='Inductive')
parser.add_argument('--dataSet', type=str, default='CiteSeer')
parser.add_argument('-e', dest="epoch_number", default=100 , help="Number of Epochs")
parser.add_argument('-mask', dest="mask", default=0, help="mask with this value during testing")
parser.add_argument('--alpha', dest="alpha", default=0, help="alpha in objective function")
parser.add_argument('--encoder_type', dest="encoder_type", default="Multi_GIN",
                    help="the encoder type, Multi_GCN, Multi_GAT, Multi_GatedGraphConv, Multi_RelGraphConv")
parser.add_argument('--loss_type', dest="loss_type", default="5", help="type of combination between loss_A and loss_F")

parser.add_argument('--b', dest="b", default='1', help="boundry for hyperparametrs")
parser.add_argument('--c', dest="c", default='1', help="number of run")
parser.add_argument('--l', dest="l", default='1', help="percentage in test edges to mask")
parser.add_argument('--model', type=str, default='KDD')

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('-num_node', dest="num_node", default=-1, type=str,
                    help="the size of subgraph which is sampled; -1 means use the whole graph")
parser.add_argument('--config', type=str, default='experiments.conf')
parser.add_argument('-decoder_type', dest="decoder_type", default="ML_SBM",
                    help="the decoder type, Either SBM or InnerDot  or TransE or MapedInnerProduct_SBM or multi_inner_product and TransX or SBM_REL")


parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-NofRels', dest="num_of_relations", default=1,
                    help="Number of latent or known relation; number of deltas in SBM")
parser.add_argument('-NofCom', dest="num_of_comunities", default=128,
                    help="Number of comunites, tor latent space dimention; len(z)")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('-DR', dest="DropOut_rate", default=.3, help="drop out rate")
parser.add_argument('-encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the size of gcn; Note: the last layer size is determine with -NofCom")
parser.add_argument('-lr', dest="lr", default=0.01, help="model learning rate")
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1,
                    help="the rate of negative samples which should be used in each epoch; by default negative sampling wont use")
parser.add_argument('-v', dest="Vis_step", default=5, help="model learning rate")
parser.add_argument('-modelpath', dest="mpath", default="VGAE_FrameWork_MODEL", type=str,
                    help="The pass to save the learned model")
parser.add_argument('-Split', dest="split_the_data_to_train_test", default=True,
                    help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-s', dest="save_embeddings_to_file", default=True, help="save the latent vector of nodes")
parser.add_argument('-CVAE_architecture', dest="CVAE_architecture", default='separate',
                    help="the possible values are sequential, separate, and transfer")
parser.add_argument('-is_prior', dest="is_prior", default=False, help="This flag is used for sampling methods")
parser.add_argument('-targets', dest="targets", default=[], help="This list is used for sampling")
parser.add_argument('--fully_inductive', dest="fully_inductive", default=False,
                    help="This flag is used if want to have dijoint transductive and inductive sets")
parser.add_argument('--sampling_method', dest="sampling_method", default="importance_sampling ", help="This var shows sampling method it could be: monte, importance_sampling, deterministic, normalized ")
parser.add_argument('--method', dest="method", default="single", help="This var shows method it could be: multi, single, subgraph")
parser.add_argument('--query_type', dest="query_type", default="class", help="This var shows method it could be: link, node class, both")

args = parser.parse_args()
fully_inductive = args.fully_inductive
if fully_inductive:
    save_recons_adj_name =args.encoder_type[-3:] + "_" + args.sampling_method + "_fully_" + args.method + "_" + args.dataSet
else:
    save_recons_adj_name = args.encoder_type[-3:] + "_" + args.sampling_method + "_semi_" + args.method + "_" + args.dataSet

print("")
print("SETING: " + str(args))

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print('Using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'cpu'

device = torch.device(device_id)



random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# %% load data
ds = args.dataSet
dataCenter = DataCenter()
dataCenter.load_dataSet(ds)
adj_list = sparse.csr_matrix(getattr(dataCenter, ds + '_adj_lists'))
features = torch.FloatTensor(getattr(dataCenter, ds + '_feats'))
labels = torch.FloatTensor(getattr(dataCenter, ds + '_labels')).to(device)

org_adj = adj_list.toarray()



#  train inductive_model
inductive_model, z_p = helper.train_PNModel(dataCenter, features.to(device),
                                         args, device)

# Split A into test and train
trainId = getattr(dataCenter, ds + '_train')
testId = getattr(dataCenter, ds + '_test')
# testId = trainId



# defining metric lists
auc_list = []
acc_list = []
ap_list = []
precision_list = []
recall_list = []
HR_list = []

auc_list_label = []
acc_list_label = []
ap_list_label = []
precision_list_label = []
recall_list_label = []
F1_list_label = []


method = args.method
if method=='multi':
    single_link = False
    multi_link = True
    multi_single_link_bl = False
elif method == 'single':
    single_link = True
    multi_link = False
    multi_single_link_bl = False

query_type = args.query_type

pred_single_link = []
true_single_link = []
targets = []

pred_single_label = []
true_single_label = []
sampling_method = args.sampling_method

if fully_inductive:
    res = org_adj.nonzero()
    index = np.where(np.isin(res[0], testId) & np.isin(res[1], trainId) | (
                np.isin(res[1], testId) & np.isin(res[0], trainId)))  # find edges that connect test to train
    i_list = res[0][index]
    j_list = res[1][index]
    org_adj[i_list, j_list] = 0  # set all the in between edges to 0

# run recognition separately for the case of single_link
std_z_recog, m_z_recog, z_recog, re_adj_recog, re_feat_recog, re_recog_labels = run_network(features, org_adj, labels, inductive_model, targets, sampling_method,
                                                            is_prior=False)


res = org_adj.nonzero()
index = np.where(np.isin(res[0], testId))  # only one node of the 2 ends of an edge needs to be in testId
idd_list = res[0][index]
neighbour_list = res[1][index]
sample_list = random.sample(range(0, len(idd_list)), 100)

# run prior network separately
counter = 0
for i in sample_list:
    print(counter)
    counter+= 1
    targets = []
    idd = idd_list[i]
    neighbour_id = neighbour_list[i]
    adj_list_copy = copy.deepcopy(org_adj)
    neigbour_prob_single = 1
    if single_link:

        adj_list_copy = copy.deepcopy(org_adj)
        if query_type == "link" or query_type == "both":
            adj_list_copy[idd, neighbour_id] = 0  # find a test edge and set it to 0
            adj_list_copy[neighbour_id, idd] = 0  # find a test edge and set it to 0

        targets.append(idd)
        targets.append(neighbour_id)


        # run prior

        std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features, adj_list_copy, labels, inductive_model,
                                                                    targets, sampling_method,  is_prior=True)
        re_adj_prior_sig = torch.sigmoid(re_adj_prior)
        re_label_prior_sig = torch.sigmoid(re_prior_labels)
        pred_single_link.append(re_adj_prior_sig[idd, neighbour_id].tolist())
        true_single_link.append(org_adj[idd, neighbour_id].tolist())
        pred_single_label.append(re_label_prior_sig[idd])
        true_single_label.append(labels[idd])



    # if multi_link:
    #     adj_list_copy_1 = copy.deepcopy(org_adj)
    #     # if we want to set all potential edges to 1
    #     if fully_inductive: # only set the edges among the idd and other test nodes to 1
    #         adj_list_copy_1[idd, testId] = 1
    #         adj_list_copy_1[testId, idd] = 1
    #     else: # set all edges among the idd and other nodes to 1
    #         adj_list_copy_1[idd, :] = 1
    #         adj_list_copy_1[:, idd] = 1
    #
    #
    #
    #     # run recoginition to update mq and sq
    #     std_z_recog, m_z_recog, z_recog, re_adj_recog = run_network(features, adj_list_copy_1, inductive_model, [], sampling_method,
    #                                                                 is_prior=False)
    #
    #
    #
    #     true_multi_links = org_adj[idd].nonzero()
    #     false_multi_links = np.array(random.sample(list(np.nonzero(org_adj[idd] == 0)[0]), len(true_multi_links[0])))
    #
    #     target_list = [[idd, i] for i in list(true_multi_links[0])]
    #     target_list.extend([[idd, i] for i in list(false_multi_links)])
    #     target_list = np.array(target_list)
    #
    #
    #     targets = list(true_multi_links[0])
    #     targets.extend(list(false_multi_links))
    #     targets.append(idd)
    #
    #
    #     # # run prior
    #     # if the selected method is monte, this would be (all 0 + MC) or (MC) and if the selected method is IS, this would be IS
    #     adj_list_copy = copy.deepcopy(org_adj)
    #     adj_list_copy[idd, :] = 0  # set all the neigbours to 0
    #     adj_list_copy[:, idd]  = 0  # set all the neigbours to 0
    #     std_z_prior, m_z_prior, z_prior, re_adj_prior = run_network(features, adj_list_copy, inductive_model,
    #                                                                 targets, sampling_method, is_prior=True)
    #
    #
    #
    #     auc, val_acc, val_ap, precision, recall, HR = get_metrics(target_list, org_adj, re_adj_prior)
    #     auc_list.append(auc)
    #     acc_list.append(val_acc)
    #     ap_list.append(val_ap)
    #     precision_list.append(precision)
    #     recall_list.append(recall)
    #     HR_list.append(HR)



# consider negative edges for single link
if single_link:
    false_count = len(pred_single_link)
    res = np.argwhere(org_adj == 0)
    np.random.shuffle(res)
    index = np.where(np.isin(res[:, 0], testId))  # only one node of the 2 ends of an edge needs to be in testId
    test_neg_edges = res[index]
    for test_neg_edge in test_neg_edges[:false_count]:
        targets = []
        idd = test_neg_edge[0]
        neighbour_id = test_neg_edge[1]
        adj_list_copy = copy.deepcopy(org_adj)
        adj_list_copy[idd, neighbour_id] = 0
        adj_list_copy[neighbour_id, idd] = 0
        targets.append(idd)
        targets.append(neighbour_id)


        # to update mq and sq for the case of importance_sampling

        std_z_recog, m_z_recog, z_recog, re_adj_recog, re_feat_recog, re_recog_labels = run_network(features, adj_list_copy, labels, inductive_model,
                                                                    targets, sampling_method,  is_prior=False)

        std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features, org_adj, labels, inductive_model, targets,sampling_method,
                                                                    is_prior=True)

        re_adj_prior_sig = torch.sigmoid(re_adj_prior)
        re_label_prior_sig = torch.sigmoid(re_prior_labels)
        pred_single_link.extend([re_adj_prior_sig[idd, neighbour_id].tolist()])
        true_single_link.extend([org_adj[idd, neighbour_id].tolist()])
        pred_single_label.extend([re_label_prior_sig[idd]])
        true_single_label.extend([labels[idd]])


        # re_adj_recog_sig = torch.sigmoid(re_adj_recog)
        # pred_single_link.extend(re_adj_recog_sig[test_neg_edges[:false_count, 0], test_neg_edges[:false_count, 1]].tolist())
        # true_single_link.extend(org_adj[test_neg_edges[:false_count, 0], test_neg_edges[:false_count, 1]].tolist())


    auc, val_acc, val_ap, precision, recall, HR = roc_auc_single(pred_single_link, true_single_link)
    auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = roc_auc_estimator_labels(pred_single_label, true_single_label)
    auc_list.append(auc)
    acc_list.append(val_acc)
    ap_list.append(val_ap)
    precision_list.append(precision)
    recall_list.append(recall)
    HR_list.append(HR)

    auc_list_label.append(auc_l)
    acc_list_label.append(acc_l)
    ap_list_label.append(ap_l)
    precision_list_label.append(precision_l)
    recall_list_label.append(recall_l)
    F1_list_label.append(F1_score)


# Print results
if fully_inductive:
    save_recons_adj_name =args.encoder_type[-3:] + "_" + args.sampling_method + "_fully_" + args.method + "_" + args.dataSet
else:
    save_recons_adj_name = args.encoder_type[-3:] + "_" + args.sampling_method + "_semi_" + args.method + "_" + args.dataSet
print(save_recons_adj_name)
print("Link Prediction")
print("auc= %.3f , acc= %.3f ap= %.3f , precision= %.3f , recall= %.3f , HR= %.3f" %(statistics.mean(auc_list), statistics.mean(acc_list), statistics.mean(ap_list), statistics.mean(precision_list), statistics.mean(recall_list), statistics.mean(HR_list)))
print("Node Classification")
print("auc= %.3f , acc= %.3f ap= %.3f , precision= %.3f , recall= %.3f , F1_Score= %.3f" %(statistics.mean(auc_list_label), statistics.mean(acc_list_label), statistics.mean(ap_list_label), statistics.mean(precision_list_label), statistics.mean(recall_list_label), statistics.mean(F1_list_label)))

with open('./results_csv/results.csv', 'a', newline="\n") as f:
    writer = csv.writer(f)
    writer.writerow([save_recons_adj_name, statistics.mean(auc_list), statistics.mean(acc_list), statistics.mean(ap_list), statistics.mean(precision_list), statistics.mean(recall_list), statistics.mean(HR_list)])
with open('./results_csv/results.csv', 'a', newline="\n") as f:
    writer = csv.writer(f)
    writer.writerow(["labels_" + save_recons_adj_name, statistics.mean(auc_list_label), statistics.mean(acc_list_label), statistics.mean(ap_list_label), statistics.mean(precision_list_label), statistics.mean(recall_list_label), statistics.mean(F1_list_label)])
