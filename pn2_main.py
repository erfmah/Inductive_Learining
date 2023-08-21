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
#from graph_samplers import *
from models import *
import plotter as plotter
import graph_statistics as GS
import pn2_helper as helper
import classification
import statistics
import warnings
from functools import partial

warnings.simplefilter('ignore')

# %%  arg setup

##################################################################


parser = argparse.ArgumentParser(description='Inductive')
parser.add_argument('--dataSet', type=str, default='computers')
parser.add_argument('-e', dest="epoch_number", default=100, help="Number of Epochs")
parser.add_argument('-mask', dest="mask", default=0, help="mask with this value during testing")
parser.add_argument('--alpha', dest="alpha", default=0, help="alpha in objective function")
parser.add_argument('--encoder_type', dest="encoder_type", default="Multi_GAT",
                    help="the encoder type, Multi_GCN, Multi_GAT, Multi_GatedGraphConv, Multi_RelGraphConv")
parser.add_argument('--loss_type', dest="loss_type", default="7", help="type of combination between loss_A and loss_F")
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
parser.add_argument('--disjoint_transductive_inductive', dest="disjoint_transductive_inductive", default=True,
                    help="This flag is used if want to have dijoint transductive and inductive sets")
parser.add_argument('--sampling_method', dest="sampling_method", default="deterministic", help="This var shows sampling method it could be: monte, importance_sampling, deterministic, normalized ")
parser.add_argument('--method', dest="method", default="multi", help="This var shows method it could be: multi, single, subgraph")



args_kdd = parser.parse_args()
save_recons_adj_name = args_kdd.loss_type+args_kdd.encoder_type+"_"
mask = int(args_kdd.mask)
l = int(args_kdd.l)
disjoint_transductive_inductive = args_kdd.disjoint_transductive_inductive
if disjoint_transductive_inductive=="False":
    disjoint_transductive_inductive = False
elif disjoint_transductive_inductive=="True":
    disjoint_transductive_inductive = True
if disjoint_transductive_inductive:
    save_recons_adj_name = save_recons_adj_name + args_kdd.sampling_method + "_fully_"
else:
    save_recons_adj_name = save_recons_adj_name + args_kdd.sampling_method + "_semi_"

print("")
print("SETING: " + str(args_kdd))



pltr = plotter.Plotter(functions=["Accuracy", "loss", "AUC"])

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print('Using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'cpu'

device = torch.device(device_id)

# %% load config

random.seed(args_kdd.seed)
np.random.seed(args_kdd.seed)
torch.manual_seed(args_kdd.seed)
torch.cuda.manual_seed_all(args_kdd.seed)

# load config file
config = pyhocon.ConfigFactory.parse_file(args_kdd.config)

# %% load data
ds = args_kdd.dataSet
dataCenter_kdd = DataCenter(config)
dataCenter_kdd.load_dataSet(ds, "KDD")
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds + '_feats')).to(device)
labels = torch.FloatTensor(getattr(dataCenter_kdd, ds + '_labels')).to(device)
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds + '_adj_lists'))


# %%  train inductive_pn
inductive_pn, z_p = helper.train_PNModel(dataCenter_kdd, features_kdd,
                                         args_kdd, device)

# Split A into test and train
trainId = getattr(dataCenter_kdd, ds + '_train')
testId = getattr(dataCenter_kdd, ds + '_test')
validId = getattr(dataCenter_kdd, ds + '_val')
labels = getattr(dataCenter_kdd, ds + '_labels')
adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds + '_adj_lists'))
# test_edges_false, test_edges, train_edges_false, train_edges, val_edes_false, val_edges = mask_test_edges(adj_list,
#                                                                                                           testId,
#                                                                                                           trainId,
#                                                                                                           validId)

auc_list_multi = []
val_acc_list_multi = []
val_ap_list_multi = []
precision_list_multi = []
recall_list_multi = []
CVAE_list_multi = []
HR_list_multi = []
CLL_list_multi = []
neighbour_prob_multi_link_list = []

auc_list_subgraph = []
val_acc_list_subgraph= []
val_ap_list_subgraph = []
precision_list_subgraph = []
recall_list_subgraph = []
CVAE_list_subgraph = []
HR_list_subgraph = []
CLL_list_subgraph = []
subgraph_prob_subgraph_link_list = []

auc_list_single = []
val_acc_list_single = []
val_ap_list_single = []
precision_list_single = []
recall_list_single = []
HR_list_single = []
CVAE_list_single = []
CLL_list_single = []

auc_list_feat_single = []
val_acc_list_feat_single = []
val_ap_list_feat_single = []
precision_list_feat_single = []
recall_list_feat_single = []
HR_list_feat_single = []
CVAE_list_feat_single = []
CLL_list_feat_single = []

auc_list_feat_multi = []
val_acc_list_feat_multi = []
val_ap_list_feat_multi = []
precision_list_feat_multi = []
recall_list_feat_multi = []
HR_list_feat_multi = []
CVAE_list_feat_multi = []
CLL_list_feat_multi = []

auc_list_feat_subgraph = []
val_acc_list_feat_subgraph = []
val_ap_list_feat_subgraph = []
precision_list_feat_subgraph = []
recall_list_feat_subgraph = []
HR_list_feat_subgraph = []
CVAE_list_feat_subgraph = []
CLL_list_feat_subgraph = []

auc_list_multi_single = []
val_acc_list_multi_single = []
val_ap_list_multi_single = []
precision_list_multi_single = []
recall_list_multi_single = []
HR_list_multi_single = []
CVAE_list_multi_single = []
CLL_list_multi_single = []

auc_list_labels_single = []
val_acc_list_labels_single = []
val_ap_list_labels_single = []
precision_list_labels_single = []
recall_list_labels_single = []
HR_list_labels_single = []
CVAE_list_labels_single = []
CLL_list_labels_single = []

auc_list_labels_multi = []
val_acc_list_labels_multi = []
val_ap_list_labels_multi = []
precision_list_labels_multi = []
recall_list_labels_multi = []
HR_list_labels_multi = []
CVAE_list_labels_multi = []
CLL_list_labels_multi = []

adj_list = sparse.csr_matrix(getattr(dataCenter_kdd, ds + '_adj_lists'))
features_kdd = torch.FloatTensor(getattr(dataCenter_kdd, ds + '_feats'))
org_adj = adj_list.toarray()

prior_only = False
method = args_kdd.method
if method=='multi':
    single_link = False
    multi_link = True
    multi_single_link_bl = False
    subgraph = False
elif method == 'single':
    single_link = True
    multi_link = False
    multi_single_link_bl = False
    subgraph = False
elif method == 'subgraph':
    single_link = False
    multi_link = False
    multi_single_link_bl = False
    subgraph = True
else:
    single_link = False
    multi_link = False
    multi_single_link_bl = True
    subgraph = False

if multi_link:
    save_recons_adj_name = save_recons_adj_name + str(l)+"_"+'multi_'+ds
elif single_link:
    save_recons_adj_name = save_recons_adj_name + str(l)+"_"+'single_'+ds
elif subgraph:
    save_recons_adj_name = save_recons_adj_name + str(l)+"_"+'subgraph_' + ds
else:
    save_recons_adj_name = save_recons_adj_name + str(l)+"_"+'multi_link_'+ds



pred_single_link = []
true_single_link = []

pred_feat_single_link = []
true_feat_single_link = []
target_feat_index_node = []

pred_multi_single_link = []
true_multi_single_link = []
pred_multi_link = []
true_multi_link = []





targets = []
sampling_method = args_kdd.sampling_method

if disjoint_transductive_inductive:
    res = org_adj.nonzero()
    index = np.where(np.isin(res[0], testId) & np.isin(res[1], trainId) | (
                np.isin(res[1], testId) & np.isin(res[0], trainId)))  # find edges that connect test to train
    i_list = res[0][index]
    j_list = res[1][index]
    org_adj[i_list, j_list] = 0  # set all the in between edges to 0

# run recognition separately
std_z_recog, m_z_recog, z_recog, re_adj_recog, reconstructed_feat, reconstructed_labels = run_network(features_kdd, org_adj, labels, inductive_pn, targets, sampling_method,
                                                            is_prior=False)
re_adj_recog_sig = torch.sigmoid(re_adj_recog)
# run prior network separately
res = org_adj.nonzero()
index = np.where(np.isin(res[0], testId))  # only one node of the 2 ends of an edge needs to be in testId
idd_list = res[0][index]
neighbour_list = res[1][index]
sample_list = random.sample(range(0, len(idd_list)), 100)
false_multi_links_list = []

save_recons_adj_name = args_kdd.c+"_"+args_kdd.b + "_"+save_recons_adj_name
with open ('./results_csv/results_CLL.csv','w') as f:
    wtr = csv.writer(f)
    wtr.writerow(['','q'])

with open ('results_csv/loss_adj_train.csv', 'a') as f:
    wtr = csv.writer(f)
    wtr.writerow([save_recons_adj_name])
with open ('results_csv/loss_feat_train.csv', 'a') as f:
    wtr = csv.writer(f)
    wtr.writerow([save_recons_adj_name])
with open ('results_csv/loss_train.csv', 'a') as f:
    wtr = csv.writer(f)
    wtr.writerow([save_recons_adj_name])
with open ('results_csv/loss_val.csv', 'a') as f:
    wtr = csv.writer(f)
    wtr.writerow([save_recons_adj_name])
xx = 0

counter = 0
for i in sample_list:
    print(xx)
    xx +=1
    save_recons_adj_name_i = save_recons_adj_name + '_' + str(i)
    targets = []
    target_feat_index_node = []
    target_feat_node_id = []
    target_feat_1 = []
    target_feat_2 = []
    idd = idd_list[i]
    neighbour_id = neighbour_list[i]
    adj_list_copy = copy.deepcopy(org_adj)
    neigbour_prob_single = 1
    if features_kdd[idd].nonzero().shape[0] > 0:
        counter += 1
        if single_link:
            #print(idd, neighbour_id)
            adj_list_copy = copy.deepcopy(org_adj)
            feat_list_copy = copy.deepcopy(features_kdd)
            adj_list_copy[idd, neighbour_id] = 0 # find a test edge and set it to 0
            adj_list_copy[neighbour_id, idd] = 0  # find a test edge and set it to 0

            #setting feature to 0 and sampling equal number of 1s and 0s
            feat_ones = np.argwhere(feat_list_copy[idd] == 1).tolist()[0]
            target_feat_1.extend(feat_ones)
            feat_zero = np.argwhere(feat_list_copy[idd] == 0)
            zero_feat_index = np.random.choice(feat_zero[0], len(feat_list_copy[idd].nonzero())).tolist()
            target_feat_1.extend(zero_feat_index)
            feat_list_copy[idd, target_feat_1] = mask

            target_feat_index_node.append(target_feat_1)
            target_feat_node_id.append(idd)

            #second node of the edge

            feat_ones = np.argwhere(feat_list_copy[neighbour_id] == 1).tolist()[0]
            target_feat_2.extend(feat_ones)
            feat_zero = np.argwhere(feat_list_copy[neighbour_id] == 0)
            zero_feat_index = np.random.choice(feat_zero[0], len(feat_list_copy[neighbour_id].nonzero())).tolist()
            target_feat_2.extend(zero_feat_index)
            feat_list_copy[neighbour_id, target_feat_2] = mask
            target_feat_index_node.append(target_feat_2)

            # target_feat_node_id.append(neighbour_id)

            #for important sampling and monte carlo
            targets.append(idd)
            targets.append(neighbour_id)

            test_adj = torch.zeros(adj_list_copy.shape[0],adj_list_copy.shape[0])
            std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(feat_list_copy, test_adj, labels, inductive_pn,
                                                                        targets, sampling_method,  is_prior=True)
            if prior_only:
                CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            else:
                CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            CVAE_list_single.append(CVAE)
            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            re_feat_prior_sig = torch.sigmoid(re_feat_prior)
            pred_single_link.append(re_adj_prior_sig[idd, neighbour_id].tolist())
            true_single_link.append(org_adj[idd, neighbour_id].tolist())
            # pred_single_link.append(re_adj_prior_sig[id_neg, neighbour_id_neg].tolist())
            # true_single_link.append(org_adj[id_neg, neighbour_id_neg].tolist())
            # true_feat_single_link.append(features_kdd[idd].tolist())
            #torch.save(re_adj_prior, './output_csv/'+save_recons_adj_name+'/'+save_recons_adj_name_i+'.pt')
            auc_feat, val_acc_feat, val_ap_feat, precision_feat, recall_feat, HR_feat, CLL_feat = roc_auc_estimator_feat(target_feat_index_node, target_feat_node_id,
                                                                                                                         re_feat_prior_sig,
                                                                                                                         features_kdd)
            auc_labels, val_acc_labels, val_ap_labels, precision_labels, recall_labels, HR_labels, CLL_labels = roc_auc_estimator_labels(targets, re_prior_labels, labels)

            auc_list_labels_single.append(auc_labels)
            val_acc_list_labels_single.append(val_acc_labels)
            val_ap_list_labels_single.append(val_ap_labels)
            precision_list_labels_single.append(precision_labels)
            recall_list_labels_single.append(recall_labels)
            HR_list_labels_single.append(HR_labels)
            CLL_list_labels_single.append(CLL_labels)

            auc_list_feat_single.append(auc_feat)
            val_acc_list_feat_single.append(val_acc_feat)
            val_ap_list_feat_single.append(val_ap_feat)
            precision_list_feat_single.append(precision_feat)
            recall_list_feat_single.append(recall_feat)
            HR_list_feat_single.append(HR_feat)
            CLL_list_feat_single.append(CLL_feat)



        if multi_link:
            adj_list_copy = copy.deepcopy(org_adj)
            feat_list_copy = copy.deepcopy(features_kdd)

            adj_list_copy[idd, :] = 0  # set all the neigbours to 0
            adj_list_copy[:, idd] = 0  # set all the neigbours to 0

            true_multi_links = org_adj[idd].nonzero()
            #selecting indexes from idd feature matrix and set them to 0
            feat_ones = np.argwhere(feat_list_copy[idd] == 1).tolist()[0]
            target_feat_1.extend(feat_ones)
            target_feat_1.extend(feat_ones)
            feat_zero = np.argwhere(feat_list_copy[idd] == 0)
            zero_feat_index = np.random.choice(feat_zero[0], len(feat_list_copy[idd].nonzero())).tolist()
            target_feat_1.extend(zero_feat_index)
            feat_list_copy[idd] = mask

            target_feat_index_node.append(target_feat_1)
            target_feat_node_id.append(idd)

            l = math.ceil(len(true_multi_links)/l)
            for neighbour in true_multi_links[0][:l]:
                feat_ones = np.argwhere(feat_list_copy[neighbour] == 1).tolist()[0]
                target_feat_2.extend(feat_ones)
                feat_zero = np.argwhere(feat_list_copy[neighbour] == 0)
                zero_feat_index = np.random.choice(feat_zero[0],
                                                   len(feat_list_copy[neighbour].nonzero())).tolist()
                target_feat_2.extend(zero_feat_index)
                feat_list_copy[neighbour, target_feat_2] = mask
                target_feat_index_node.append(target_feat_2)
                target_feat_node_id.append(neighbour)


            false_multi_link = np.array(random.sample(list(np.nonzero(org_adj[idd] == 0)[0]), len(true_multi_links[0])))
            for j in list(false_multi_link)[:l]:
                false_multi_links_list.append([idd, i])
                adj_list_copy[j, idd] = 0
                adj_list_copy[idd, j] = 0
                feat_ones = np.argwhere(feat_list_copy[j] == 1).tolist()[0]
                target_feat_2.extend(feat_ones)
                feat_zero = np.argwhere(feat_list_copy[j] == 0)
                zero_feat_index = np.random.choice(feat_zero[0],
                                                   len(feat_list_copy[j].nonzero())).tolist()
                target_feat_2.extend(zero_feat_index)
                feat_list_copy[j, target_feat_2] = 2
                target_feat_index_node.append(target_feat_2)
                target_feat_node_id.append(j)

            target_list = [[idd, i] for i in list(true_multi_links[0])]
            target_list.extend([[idd, i] for i in list(false_multi_link)])

            targets = list(true_multi_links[0])
            targets.extend(list(false_multi_link)) ################################ add back for importance sampling
            targets.append(idd)

            std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_labels_prior = run_network(feat_list_copy, adj_list_copy, labels, inductive_pn,
                                                                        targets, sampling_method, is_prior=True)

            if prior_only:
                CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            else:
                CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            CVAE_list_multi.append(CVAE)
            re_feat_prior_sig = torch.sigmoid(re_feat_prior)
            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            target_list = np.array(target_list)
            # pred_multi_link.extend(re_adj_prior_sig[target_list[:, 0], target_list[:, 1]].tolist())
            # true_multi_link.extend(org_adj[target_list[:, 0], target_list[:, 1]].tolist())
            auc_labels, val_acc_labels, val_ap_labels, precision_labels, recall_labels, HR_labels, CLL_labels = roc_auc_estimator_labels(
                targets, re_labels_prior, labels)

            auc_list_labels_multi.append(auc_labels)
            val_acc_list_labels_multi.append(val_acc_labels)
            val_ap_list_labels_multi.append(val_ap_labels)
            precision_list_labels_multi.append(precision_labels)
            recall_list_labels_multi.append(recall_labels)
            HR_list_labels_multi.append(HR_labels)
            CLL_list_labels_multi.append(CLL_labels)

            auc, val_acc, val_ap, precision, recall, HR, CLL = get_metrices(target_list, org_adj, re_adj_prior)
            auc_list_multi.append(auc)
            val_acc_list_multi.append(val_acc)
            val_ap_list_multi.append(val_ap)
            precision_list_multi.append(precision)
            recall_list_multi.append(recall)
            HR_list_multi.append(HR)
            CLL_list_multi = CLL

            auc_feat, val_acc_feat, val_ap_feat, precision_feat, recall_feat, HR_feat, CLL_feat = roc_auc_estimator_feat(
                target_feat_index_node, target_feat_node_id,
                re_feat_prior_sig,
                features_kdd)
            auc_list_feat_multi.append(auc_feat)
            val_acc_list_feat_multi.append(val_acc_feat)
            val_ap_list_feat_multi.append(val_ap_feat)
            precision_list_feat_multi.append(precision_feat)
            recall_list_feat_multi.append(recall_feat)
            HR_list_feat_multi.append(HR_feat)
            CLL_list_feat_multi.append(CLL_feat)
            # with open('./results_csv/results.csv', 'a', newline="\n") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([save_recons_adj_name_i])
            #     writer.writerow([precision, recall, val_acc, val_ap, auc, CLL, HR])

            #torch.save(re_adj_prior, './output_csv/'+save_recons_adj_name_i+'.pt')

            # neighbour_prob_multi_link_list.append(get_neighbour_prob(re_adj_prior, idd, org_adj[idd].nonzero()[
            #     0]).item())  # this function calculates the prob of all positive edges around idd node



        if subgraph:
            pos_edges, neg_edges, nodes = get_subgraph_random_walk(org_adj, idd, 4)
            # print(idd, neighbour_id)
            adj_list_copy = copy.deepcopy(org_adj)
            feat_list_copy = copy.deepcopy(features_kdd)
            for e in pos_edges:
                adj_list_copy[e[0], e[1]] = 0


            feat_ones = np.argwhere(feat_list_copy[idd] == 1).tolist()[0]
            target_feat_1.extend(feat_ones)
            target_feat_1.extend(feat_ones)
            feat_zero = np.argwhere(feat_list_copy[idd] == 0)
            zero_feat_index = np.random.choice(feat_zero[0], len(feat_list_copy[idd].nonzero())).tolist()
            target_feat_1.extend(zero_feat_index)
            feat_list_copy[idd] = mask

            target_feat_index_node.append(target_feat_1)
            target_feat_node_id.append(idd)
            # setting feature to 0 and sampling equal number of 1s and 0s
            l = (math.ceil(len(pos_edges)/l))
            for neighbour in pos_edges[:l]:
                feat_list_copy[neighbour[1]] = 0
                feat_ones = np.argwhere(feat_list_copy[neighbour[1]] == 1).tolist()[0]
                target_feat_2.extend(feat_ones)
                feat_zero = np.argwhere(feat_list_copy[neighbour[1]] == 0)
                zero_feat_index = np.random.choice(feat_zero[0],
                                                   len(feat_list_copy[neighbour].nonzero())).tolist()
                target_feat_2.extend(zero_feat_index)
                # feat_list_copy[neighbour, target_feat_2] = mask
                target_feat_index_node.append(target_feat_2)
                target_feat_node_id.append(neighbour[1])

            for neighbour in neg_edges[:l]:
                feat_list_copy[neighbour[1]] = 0
                feat_ones = np.argwhere(feat_list_copy[neighbour[1]] == 1).tolist()[0]
                target_feat_2.extend(feat_ones)
                feat_zero = np.argwhere(feat_list_copy[neighbour[1]] == 0)
                zero_feat_index = np.random.choice(feat_zero[0],
                                                   len(feat_list_copy[neighbour[1]].nonzero())).tolist()
                target_feat_2.extend(zero_feat_index)
                # feat_list_copy[neighbour[1], target_feat_2] = mask
                target_feat_index_node.append(target_feat_2)
                target_feat_node_id.append(neighbour[1])
            # target_feat_node_id.append(neighbour_id)

            target_list = copy.deepcopy(pos_edges)
            target_list.extend(neg_edges)
            # for important sampling and monte carlo
            targets.append(nodes)

            std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior = run_network(feat_list_copy, adj_list_copy,
                                                                                       inductive_pn,
                                                                                       targets[0], sampling_method,
                                                                                       is_prior=True)

            if prior_only:
                CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            else:
                CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(), org_adj,
                                 idd, neighbour_id).detach().numpy()
            CVAE_list_subgraph.append(CVAE)
            re_feat_prior_sig = torch.sigmoid(re_feat_prior)
            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            target_list = np.array(target_list)
            # pred_multi_link.extend(re_adj_prior_sig[target_list[:, 0], target_list[:, 1]].tolist())
            # true_multi_link.extend(org_adj[target_list[:, 0], target_list[:, 1]].tolist())

            auc, val_acc, val_ap, precision, recall, HR, CLL = get_metrices(target_list, org_adj, re_adj_prior)
            auc_list_subgraph.append(auc)
            val_acc_list_subgraph.append(val_acc)
            val_ap_list_subgraph.append(val_ap)
            precision_list_subgraph.append(precision)
            recall_list_subgraph.append(recall)
            HR_list_subgraph.append(HR)
            CLL_list_subgraph = CLL

            auc_feat, val_acc_feat, val_ap_feat, precision_feat, recall_feat, HR_feat, CLL_feat = roc_auc_estimator_feat(
                target_feat_index_node, target_feat_node_id,
                re_feat_prior_sig,
                features_kdd)
            auc_list_feat_subgraph.append(auc_feat)
            val_acc_list_feat_subgraph.append(val_acc_feat)
            val_ap_list_feat_subgraph.append(val_ap_feat)
            precision_list_feat_subgraph.append(precision_feat)
            recall_list_feat_subgraph.append(recall_feat)
            HR_list_feat_subgraph.append(HR_feat)
            CLL_list_feat_subgraph.append(CLL_feat)

    if multi_single_link_bl:
            for nn in org_adj[idd].nonzero()[0]:
                adj_list_copy = copy.deepcopy(org_adj)
                # set all the neighbours expect the test one to 0 for recognition network
                adj_list_copy, false_count = get_single_link_evidence(adj_list_copy, idd,
                                                                      np.delete(adj_list_copy[idd].nonzero()[0],
                                                                                np.where(adj_list_copy[idd].nonzero()[
                                                                                             0] == nn)))
                std_z_recog, m_z_recog, z_recog, re_adj_recog, reconstructed_feat = run_network(features_kdd, adj_list_copy, inductive_pn,
                                                                            targets, sampling_method,
                                                                            is_prior=False)

                adj_list_copy[idd, nn] = 0  # find a test edge and set it to 0
                adj_list_copy[nn, idd] = 0  # find a test edge and set it to 0

                targets.append(idd)
                targets.append(nn)
                std_z_prior, m_z_prior, z_prior, re_adj_prior, reconstructed_feat = run_network(features_kdd, adj_list_copy, inductive_pn,
                                                                            targets, sampling_method, is_prior=True)

                if prior_only:
                    CVAE = CVAE_loss(m_z_prior, m_z_prior, std_z_prior, std_z_prior, re_adj_prior.detach().numpy(),
                                     org_adj, idd, neighbour_id).detach().numpy()
                else:
                    CVAE = CVAE_loss(m_z_recog, m_z_prior, std_z_recog, std_z_prior, re_adj_prior.detach().numpy(),
                                     org_adj, idd, neighbour_id).detach().numpy()
                CVAE_list_multi_single.append(CVAE)

                re_adj_prior_sig = torch.sigmoid(re_adj_prior)
                pred_multi_single_link.append(re_adj_prior_sig[idd, nn].tolist())
                true_multi_single_link.append(org_adj[idd, nn].tolist())

            # Get false edges
            std_z_recog, m_z_recog, z_recog, re_adj_recog, reconstructed_feat = run_network(features_kdd, org_adj, inductive_pn, targets,
                                                                         sampling_method, is_prior=False)
            re_adj_recog_sig = torch.sigmoid(re_adj_recog)
            res = np.argwhere(org_adj[idd] == 0)
            np.random.shuffle(res)
            index = np.where(np.isin(res[:, 0], testId))  # only one node of the 2 ends of an edge needs to be in testId
            test_neg_edges = res[index]
            true_multi_single_link.extend(
                org_adj[test_neg_edges[:false_count, 0], test_neg_edges[:false_count, 1]].tolist())
            auc, val_acc, val_ap, conf_mtrx, precision, recall, HR, CLL = roc_auc_single(pred_multi_single_link,
                                                                                    true_multi_single_link)
            auc_list_multi_single.append(auc)
            val_acc_list_multi_single.append(val_acc)
            val_ap_list_multi_single.append(val_ap)
            precision_list_multi_single.append(precision)
            recall_list_multi_single.append(recall)
            HR_list_multi_single.append(HR)
            CLL_list_multi.append(CLL)

if single_link:
    # false_count = len(pred_single_link)
    # res = np.argwhere(org_adj == 0)
    # np.random.shuffle(res)
    # index = np.where(np.isin(res[:, 0], testId))  # only one node of the 2 ends of an edge needs to be in testId
    # test_neg_edges = res[index]
    #
    # for test_neg_edge in test_neg_edges[:false_count]:
    #     targets = []
    #     idd = test_neg_edge[0]
    #     neighbour_id = test_neg_edge[1]
    #     adj_list_copy = copy.deepcopy(org_adj)
    #     adj_list_copy[idd, neighbour_id] = 0
    #     adj_list_copy[neighbour_id, idd] = 0
    #
    #     targets.append(idd)
    #     targets.append(neighbour_id)
    #
    #     std_z_prior, m_z_prior, z_prior, re_adj_prior, reconstructed_feat = run_network(features_kdd, adj_list_copy, inductive_pn,
    #                                                                 targets, sampling_method,  is_prior=True)
    # re_adj_prior_sig = torch.sigmoid(re_adj_prior)
    # pred_single_link.append(re_adj_prior_sig[idd, neighbour_id].tolist())
    # true_single_link.append(org_adj[idd, neighbour_id].tolist())
    #
    #
    #
    #
    # auc, val_acc, val_ap, precision, recall, HR, CLL = roc_auc_single(pred_single_link, true_single_link)
    # auc_list_single.append(auc)
    # val_acc_list_single.append(val_acc)
    # val_ap_list_single.append(val_ap)
    # precision_list_single.append(precision)
    # recall_list_single.append(recall)
    # HR_list_single.append(HR)
    # CLL_list_single.append(CLL)
    false_count = len(pred_single_link)
    res = np.argwhere(org_adj == 0)
    np.random.shuffle(res)
    index = np.where(np.isin(res[:, 0], testId))  # only one node of the 2 ends of an edge needs to be in testId
    test_neg_edges = res[index]
    re_adj_recog_sig = torch.sigmoid(re_adj_recog)
    #adding feature prediction
    neg_edges = test_neg_edges[:false_count]
    counter = 100
    for idd, neighbour_id in neg_edges:
        print(counter)
        counter += 1
        adj_list_copy = copy.deepcopy(org_adj)
        feat_list_copy = copy.deepcopy(features_kdd)

        # setting feature to 0 and sampling equal number of 1s and 0s
        feat_ones = np.argwhere(feat_list_copy[idd] == 1).tolist()[0]
        target_feat_1.extend(feat_ones)
        feat_zero = np.argwhere(feat_list_copy[idd] == 0)
        zero_feat_index = np.random.choice(feat_zero[0], len(feat_list_copy[idd].nonzero())).tolist()
        target_feat_1.extend(zero_feat_index)
        feat_list_copy[idd, target_feat_1] = mask


        target_feat_index_node.append(target_feat_1)
        target_feat_node_id.append(idd)

        #for the second node of the edge

        feat_ones = np.argwhere(feat_list_copy[neighbour_id] == 1).tolist()[0]
        target_feat_2.extend(feat_ones)
        feat_zero = np.argwhere(feat_list_copy[neighbour_id] == 0)
        zero_feat_index = np.random.choice(feat_zero[0], len(feat_list_copy[neighbour_id].nonzero())).tolist()
        target_feat_2.extend(zero_feat_index)
        feat_list_copy[neighbour_id, target_feat_2] = mask
        target_feat_index_node.append(target_feat_2)

        target_feat_node_id.append(neighbour_id)

        # end of setting features to 0

        #for important sampling
        targets.append(idd)
        targets.append(neighbour_id)
        std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(feat_list_copy, adj_list_copy, labels,
                                                                                   inductive_pn,
                                                                                   targets, sampling_method,
                                                                                   is_prior=True)
        re_adj_prior_sig = torch.sigmoid(re_adj_prior)
        re_feat_prior_sig = torch.sigmoid(re_feat_prior)
        pred_single_link.append(re_adj_prior_sig[idd, neighbour_id].tolist())
        true_single_link.append(org_adj[idd, neighbour_id].tolist())

        auc_feat, val_acc_feat, val_ap_feat, precision_feat, recall_feat, HR_feat, CLL_feat = roc_auc_estimator_feat(
            target_feat_index_node, target_feat_node_id,
            re_feat_prior_sig,
            features_kdd)

        auc_labels, val_acc_labels, val_ap_labels, precision_labels, recall_labels, HR_labels, CLL_labels = roc_auc_estimator_labels(
            targets, re_prior_labels, labels)

        auc_list_labels_single.append(auc_labels)
        val_acc_list_labels_single.append(val_acc_labels)
        val_ap_list_labels_single.append(val_ap_labels)
        precision_list_labels_single.append(precision_labels)
        recall_list_labels_single.append(recall_labels)
        HR_list_labels_single.append(HR_labels)
        CLL_list_labels_single.append(CLL_labels)


        auc_list_feat_single.append(auc_feat)
        val_acc_list_feat_single.append(val_acc_feat)
        val_ap_list_feat_single.append(val_ap_feat)
        precision_list_feat_single.append(precision_feat)
        recall_list_feat_single.append(recall_feat)
        HR_list_feat_single.append(HR_feat)
        CLL_list_feat_single.append(CLL_feat)

    #end of feature predictoin


    auc, val_acc, val_ap, precision, recall, HR, CLL = roc_auc_single(pred_single_link, true_single_link)
    # auc_feat, val_acc_feat, val_ap_feat, precision_feat, recall_feat, HR_feat, CLL_feat = get_metrices(target_feat, pred_feat_single_link, true_feat_single_link)
    auc_list_single.append(auc)
    val_acc_list_single.append(val_acc)
    val_ap_list_single.append(val_ap)
    precision_list_single.append(precision)
    recall_list_single.append(recall)
    HR_list_single.append(HR)
    CLL_list_single.append(CLL)



# only use for A0, A1
# if multi_link:
#     for false_multi_links in false_multi_links_list:
#         adj_list_copy = copy.deepcopy(org_adj)
#         targets = []
#
#         # adj_list_copy[false_multi_links[:, 0], false_multi_links[:, 1]] = 1
#         # adj_list_copy[false_multi_links[:, 1], false_multi_links[:, 0]] = 1
#         # targets = false_multi_links[:, 1].tolist()
#         # targets.append(false_multi_links[0][0])
#
#         for false_multi_link in false_multi_links:
#             idd = false_multi_link[0]
#             neighbour_id = false_multi_link[1]
#             adj_list_copy[idd, neighbour_id] = 1
#             adj_list_copy[neighbour_id, idd] = 1
#             targets.append(neighbour_id)
#         targets.append(idd)
#
#         std_z_recog, m_z_recog, z_recog, re_adj_recog = run_network(features_kdd, adj_list_copy, inductive_pn,
#                                                                     [],
#                                                                     is_prior=False)
#
#         std_z_prior, m_z_prior, z_prior, re_adj_prior = run_network(features_kdd, org_adj, inductive_pn, targets,
#                                                                     is_prior=True)
#
#         re_adj_prior_sig = torch.sigmoid(re_adj_prior)
#
#         false_multi_links = np.array(false_multi_links)
#         pred_multi_link.extend(re_adj_prior_sig[false_multi_links[:, 0], false_multi_links[:, 1]].tolist())
#         true_multi_link.extend(org_adj[false_multi_links[:, 0], false_multi_links[:, 1]].tolist())
#
#     auc, val_acc, val_ap, precision, recall, HR = roc_auc_single(pred_multi_link, true_multi_link)
#     auc_list_multi.append(auc)
#     val_acc_list_multi.append(val_acc)
#     val_ap_list_multi.append(val_ap)
#     precision_list_multi.append(precision)
#     recall_list_multi.append(recall)
#     HR_list_multi.append(HR)

# Print results
print(save_recons_adj_name)

if multi_link:
    auc_mean_multi = statistics.mean(auc_list_multi)
    val_acc_mean_multi = statistics.mean(val_acc_list_multi)
    val_ap_mean_multi = statistics.mean(val_ap_list_multi)
    precision_mean_multi = statistics.mean(precision_list_multi)
    recall_mean_multi = statistics.mean(recall_list_multi)
    HR_mean_multi = statistics.mean(HR_list_multi)
    CLL_mean_multi = statistics.mean(CLL_list_multi)

    auc_mean_feat_multi = statistics.mean(auc_list_feat_multi)
    val_acc_mean_feat_multi = statistics.mean(val_acc_list_feat_multi)
    val_ap_mean_feat_multi = statistics.mean(val_ap_list_feat_multi)
    precision_mean_feat_multi = statistics.mean(precision_list_feat_multi)
    recall_mean_feat_multi = statistics.mean(recall_list_feat_multi)
    HR_mean_feat_multi = statistics.mean(HR_list_feat_multi)
    # CLL_mean_feat_multi = statistics.mean(CLL_list_feat_multi)
    CLL_mean_feat_multi = 0

    auc_mean_labels_multi = statistics.mean(auc_list_labels_multi)
    val_acc_mean_labels_multi = statistics.mean(val_acc_list_labels_multi)
    val_ap_mean_labels_multi = statistics.mean(val_ap_list_labels_multi)
    precision_mean_labels_multi = statistics.mean(precision_list_labels_multi)
    recall_mean_labels_multi = statistics.mean(recall_list_labels_multi)
    HR_mean_labels_multi = statistics.mean(HR_list_labels_multi)
    CLL_mean_labels_multi = 0

    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            [save_recons_adj_name, auc_mean_multi, val_acc_mean_multi, val_ap_mean_multi, precision_mean_multi,
             recall_mean_multi, HR_mean_multi, CLL_mean_multi])
    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["feat_" + save_recons_adj_name, auc_mean_feat_multi, val_acc_mean_feat_multi, val_ap_mean_feat_multi,
             precision_mean_feat_multi, recall_mean_feat_multi, HR_mean_feat_multi, CLL_mean_feat_multi])
    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["labels_" + save_recons_adj_name, auc_mean_labels_multi, val_acc_mean_labels_multi,
                         val_ap_mean_labels_multi,
                         precision_mean_labels_multi, recall_mean_labels_multi, HR_mean_labels_multi,
                         CLL_mean_labels_multi])

    print("multi link")
    print("link prediction:")
    print("auc: ", auc_mean_multi)
    print("acc", val_acc_mean_multi)
    print("ap: ", val_ap_mean_multi)
    print("precision", precision_mean_multi)
    print("recall", recall_mean_multi)
    print("HR", HR_mean_multi)
    print("CLL", CLL_mean_multi)

    print("feature prediction:")

    print("auc: ", auc_mean_feat_multi)
    print("acc", val_acc_mean_feat_multi)
    print("ap: ", val_ap_mean_feat_multi)
    print("precision", precision_mean_feat_multi)
    print("recall", recall_mean_feat_multi)
    print("HR", HR_mean_feat_multi)
    print("CLL", CLL_mean_feat_multi)

    print("node classification:")

    print("auc: ", auc_mean_labels_multi)
    print("acc", val_acc_mean_labels_multi)
    print("ap: ", val_ap_mean_labels_multi)
    print("precision", precision_mean_labels_multi)
    print("recall", recall_mean_labels_multi)
    print("HR", HR_mean_labels_multi)
    print("CLL", CLL_mean_labels_multi)
if subgraph:
    auc_mean_subgraph = statistics.mean(auc_list_subgraph)
    val_acc_mean_subgraph = statistics.mean(val_acc_list_subgraph)
    val_ap_mean_subgraph = statistics.mean(val_ap_list_subgraph)
    precision_mean_subgraph = statistics.mean(precision_list_subgraph)
    recall_mean_subgraph = statistics.mean(recall_list_subgraph)
    HR_mean_subgraph= statistics.mean(HR_list_subgraph)
    CLL_mean_subgraph = statistics.mean(CLL_list_subgraph)

    auc_mean_feat_subgraph = statistics.mean(auc_list_feat_subgraph)
    val_acc_mean_feat_subgraph = statistics.mean(val_acc_list_feat_subgraph)
    val_ap_mean_feat_subgraph = statistics.mean(val_ap_list_feat_subgraph)
    precision_mean_feat_subgraph = statistics.mean(precision_list_feat_subgraph)
    recall_mean_feat_subgraph= statistics.mean(recall_list_feat_subgraph)
    HR_mean_feat_subgraph= statistics.mean(HR_list_feat_subgraph)
    # CLL_mean_feat_subgraph = statistics.mean(CLL_list_feat_subgraph)
    CLL_mean_feat_subgraph = 0
    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        # writer.writerow([save_recons_adj_name,"","","","","",""])
        writer.writerow([save_recons_adj_name, auc_mean_subgraph, val_acc_mean_subgraph, val_ap_mean_subgraph, precision_mean_subgraph, recall_mean_subgraph, HR_mean_subgraph, CLL_mean_subgraph])
        writer.writerow(["feat_"+save_recons_adj_name, auc_mean_feat_subgraph, val_acc_mean_feat_subgraph, val_ap_mean_feat_subgraph, precision_mean_feat_subgraph,
             recall_mean_feat_subgraph, HR_mean_feat_subgraph, CLL_mean_feat_subgraph])

    print("subgraph")
    print("link prediction:")
    print("auc: ", auc_mean_subgraph)
    print("acc", val_acc_mean_subgraph)
    print("ap: ", val_ap_mean_subgraph)
    print("precision", precision_mean_subgraph)
    print("recall", recall_mean_subgraph)
    print("HR", HR_mean_subgraph)
    print("CLL", CLL_mean_subgraph)

    print("feature prediction:")

    print("auc: ", auc_mean_feat_subgraph)
    print("acc", val_acc_mean_feat_subgraph)
    print("ap: ", val_ap_mean_feat_subgraph)
    print("precision", precision_mean_feat_subgraph)
    print("recall", recall_mean_feat_subgraph)
    print("HR", HR_mean_feat_subgraph)
    print("CLL", CLL_mean_feat_subgraph)


if multi_single_link_bl:
    auc_mean_multi_single = statistics.mean(auc_list_multi_single)
    val_acc_mean_multi_single = statistics.mean(val_acc_list_multi_single)
    val_ap_mean_multi_single = statistics.mean(val_ap_list_multi_single)
    precision_mean_multi_single = statistics.mean(precision_list_multi_single)
    recall_mean_multi_single = statistics.mean(recall_list_multi_single)
    HR_mean_multi_single = statistics.mean(HR_list_multi_single)
    CLL_mean_multi_single = np.mean(CLL_list_multi_single)

    print("multi link")
    print("auc: ", auc_mean_multi_single)
    print("acc", val_acc_mean_multi_single)
    print("ap: ", val_ap_mean_multi_single)
    print("precision", precision_mean_multi_single)
    print("recall", recall_mean_multi_single)
    print("HR", HR_mean_multi_single)
    print("CLL", CLL_mean_multi_single)

if single_link:
    auc_mean_single = statistics.mean(auc_list_single)
    val_acc_mean_single = statistics.mean(val_acc_list_single)
    val_ap_mean_single = statistics.mean(val_ap_list_single)
    precision_mean_single = statistics.mean(precision_list_single)
    recall_mean_single = statistics.mean(recall_list_single)
    HR_mean_single = statistics.mean(HR_list_single)
    CLL_mean_single = np.mean(CLL_list_single)

    auc_mean_feat_single = statistics.mean(auc_list_feat_single)
    val_acc_mean_feat_single = statistics.mean(val_acc_list_feat_single)
    val_ap_mean_feat_single = statistics.mean(val_ap_list_feat_single)
    precision_mean_feat_single = statistics.mean(precision_list_feat_single)
    recall_mean_feat_single = statistics.mean(recall_list_feat_single)
    HR_mean_feat_single = statistics.mean(HR_list_feat_single)
    #CLL_mean_feat_single = np.mean(CLL_list_feat_single)
    CLL_mean_feat_single = []

    auc_mean_labels_single = statistics.mean(auc_list_labels_single)
    val_acc_mean_labels_single = statistics.mean(val_acc_list_labels_single)
    val_ap_mean_labels_single = statistics.mean(val_ap_list_labels_single)
    precision_mean_labels_single = statistics.mean(precision_list_labels_single)
    recall_mean_labels_single = statistics.mean(recall_list_labels_single)
    HR_mean_labels_single = statistics.mean(HR_list_labels_single)
    #CLL_mean_labels_single = np.mean(CLL_list_labels_single)
    CLL_mean_labels_single = []

    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow([save_recons_adj_name,auc_mean_single,val_acc_mean_single,val_ap_mean_single,precision_mean_single,recall_mean_single,HR_mean_single,CLL_mean_single])
    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["feat_"+save_recons_adj_name,auc_mean_feat_single,val_acc_mean_feat_single,val_ap_mean_feat_single,precision_mean_feat_single,recall_mean_feat_single,HR_mean_feat_single,CLL_mean_feat_single])
    with open('./results_csv/results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["labels_" + save_recons_adj_name, auc_mean_labels_single, val_acc_mean_labels_single, val_ap_mean_labels_single,
             precision_mean_labels_single, recall_mean_labels_single, HR_mean_labels_single, CLL_mean_labels_single])

    print("single link")
    print("auc: ", auc_mean_single)
    print("acc", val_acc_mean_single)
    print("ap: ", val_ap_mean_single)
    print("precision", precision_mean_single)
    print("recall", recall_mean_single)
    print("HR", HR_mean_single)
    print("CLL", CLL_mean_single)

    print("\nsingle link features")
    print("auc_feat: ", auc_mean_feat_single)
    print("acc_feat", val_acc_mean_feat_single)
    print("ap_feat: ", val_ap_mean_feat_single)
    print("precision_feat", precision_mean_feat_single)
    print("recall_feat", recall_mean_feat_single)
    print("HR_feat", HR_mean_feat_single)
    print("CLL_feat", CLL_mean_feat_single)

    print("\nsingle link labels")
    print("auc_labels: ", auc_mean_labels_single)
    print("acc_labels", val_acc_mean_labels_single)
    print("ap_labels: ", val_ap_mean_labels_single)
    print("precision_labels", precision_mean_labels_single)
    print("recall_labels", recall_mean_labels_single)
    print("HR_labels", HR_mean_labels_single)
    print("CLL_labels", CLL_mean_labels_single)

# print("CLL single",statistics.mean(neigbour_prob_single_list))
# print("CLL multi",statistics.mean(neighbour_prob_multi_link_list))
