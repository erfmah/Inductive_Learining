#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:13:06 2021

@author: pnaddaf
"""

import sys
import os
import argparse

import numpy as np
from scipy.sparse import lil_matrix
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import random

from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv

from dataCenter import *
from utils import *
from models import *
import timeit
#import src.plotter as plotter
#import src.graph_statistics as GS

import classification
import plotter
from bayes_opt import BayesianOptimization
from functools import partial
from scipy.optimize import minimize
import scipy.optimize as opt
from torchmetrics.classification import AUROC
from sklearn.metrics import roc_auc_score, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real
from skopt import gp_minimize

#%% KDD model
def train_PNModel(dataCenter, features, args, device):
    up_bound = int(args.b)
    loss_type = args.loss_type
    decoder = args.decoder_type
    encoder = args.encoder_type
    num_of_relations = args.num_of_relations  # diffrent type of relation
    num_of_comunities = args.num_of_comunities  # number of comunities
    batch_norm = args.batch_norm
    DropOut_rate = args.DropOut_rate
    encoder_layers = [int(x) for x in args.encoder_layers.split()]
    split_the_data_to_train_test = args.split_the_data_to_train_test
    epoch_number = args.epoch_number
    negative_sampling_rate = args.negative_sampling_rate
    visulizer_step = args.Vis_step
    PATH = args.mpath
    subgraph_size = args.num_node
    use_feature = args.use_feature
    lr = args.lr
    dataset = args.dataSet
    save_embeddings_to_file = args.save_embeddings_to_file
    is_prior = args.is_prior
    targets = args.targets
    sampling_method = args.sampling_method
    pltr = plotter.Plotter(functions=["loss",  "Accuracy", "Recons Loss", "KL", "AUC"]) 
    synthesis_graphs = {"grid", "community", "lobster", "ego"}
    alpha = args.alpha


    ds = args.dataSet
    if ds in synthesis_graphs:
        synthetic = True
    else: 
        synthetic = False
    
    original_adj_full = torch.FloatTensor(getattr(dataCenter, ds+'_adj_lists')).to(device)

    node_label_full = torch.FloatTensor(getattr(dataCenter, ds+'_labels')).to(device)
    # torch.Size([3480, 121])
    # if edge labels exist
    edge_labels = None
    if ds == 'IMDB' or ds == 'ACM' or ds == 'DBLP':
        edge_labels= torch.FloatTensor(getattr(dataCenter, ds+'_edge_labels')).to(device)
    circles = None
    
    
   
    
    # shuffling the data, and selecting a subset of it
    if subgraph_size == -1:
        subgraph_size = original_adj_full.shape[0]
    elemnt = min(original_adj_full.shape[0], subgraph_size)
    indexes = list(range(original_adj_full.shape[0]))
    np.random.shuffle(indexes)
    indexes = indexes[:elemnt]
    original_adj = original_adj_full[indexes, :]
    original_adj = original_adj[:, indexes]
    features = features[indexes]

    number_of_classes = len(node_label_full[0])
    if synthetic != True:
        if node_label_full != None:
            node_label = [np.array(node_label_full[i], dtype=np.float16) for i in indexes]
        if edge_labels != None:
            edge_labels = edge_labels[indexes, :]
            edge_labels = edge_labels[:, indexes]
        if circles != None:
            shuffles_cir = {}
            for ego_node, circule_lists in circles.items():
                shuffles_cir[indexes.index(ego_node)] = [[indexes.index(x) for x in circule_list] for circule_list in
                                                         circule_lists]
            circles = shuffles_cir
    # Check for Encoder and redirect to appropriate function
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
        # encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_GCN_2":
        encoder_model = multi_layer_GCN_2(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_GAT":
        encoder_model = multi_layer_GAT(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_GAT_2":
        encoder_model = multi_layer_GAT_2(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_GIN":
        encoder_model = multi_layer_GIN(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)
    elif encoder == "Multi_RelGraphConv":
        encoder_model = multi_layer_RelGraphConv(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
    
    elif encoder == "Multi_GatedGraphConv":
        encoder_model = multi_layer_GatedGraphConv(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "mixture_of_GCNs":
        encoder_model = mixture_of_GCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                        latent_dim=num_of_comunities, layers=encoder_layers, DropOut_rate=DropOut_rate)

    elif encoder == "mixture_of_GatedGCNs":
        encoder_model = mixture_of_GatedGCNs(in_feature=features.shape[1], num_relation=num_of_relations,
                                             latent_dim=num_of_comunities, layers=encoder_layers, dropOutRate=DropOut_rate)
    elif encoder == "Edge_GCN":
        haveedge = True
        encoder_model = edge_enabled_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)
    # asakhuja End
    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")
    
    # Check for Decoder and redirect to appropriate function
    if decoder == "SBM":
        decoder_model = SBM_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "ML_SBM":
        decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate=0.3)

    elif decoder == "multi_inner_product":
        decoder_model = MapedInnerProductDecoder([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
    
    elif decoder == "MapedInnerProduct_SBM":
        decoder_model = MapedInnerProduct_SBM([32, 32], num_of_relations, num_of_comunities, batch_norm, DropOut_rate)
    
    elif decoder == "TransE":
        decoder_model = TransE_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "TransX":
        decoder_model = TransX_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "SBM_REL":
        haveedge = True
        decoder_model = edge_enabeled_SBM_decoder(num_of_comunities, num_of_relations)
    
    elif decoder == "InnerDot":
        decoder_model = InnerProductDecoder()
    else:
        raise Exception("Sorry, this Decoder is not Impemented; check the input args")
        
    feature_decoder = feature_decoder_nn(features.shape[1], num_of_comunities)
    feature_encoder_model = feature_encoder_nn(features.view(-1, features.shape[1]), num_of_comunities)

    class_decoder = MulticlassClassifier(number_of_classes, num_of_comunities)
    if use_feature == False:
        features = torch.eye(features.shape[0])
        features = sp.csr_matrix(features)
    
    
    
    if split_the_data_to_train_test == True:
        trainId = getattr(dataCenter, ds + '_train')
        validId = getattr(dataCenter, ds + '_val')
        testId = getattr(dataCenter, ds + '_test')
        adj_train =  original_adj.cpu().detach().numpy()[trainId, :][:, trainId]
        adj_val = original_adj.cpu().detach().numpy()[validId, :][:, validId]
        
        feat_np = features.cpu().data.numpy()
        feat_train = feat_np[trainId, :]
        feat_val = feat_np[validId, :]
        
        labels_np = np.array(node_label, dtype=np.float16)
        labels_train = labels_np[trainId]
        labels_val = labels_np[validId]


        # adj_train , adj_val, adj_test, feat_train, feat_val, feat_test, train_true, train_false, val_true, val_false= make_test_train_gpu(
        #                 original_adj.cpu().detach().numpy(), features,
        #                 [trainId, validId, testId])
        print('Finish spliting dataset to train and test. ')
    
    
    
    adj_train = sp.csr_matrix(adj_train)
    adj_train = torch.tensor(adj_train.todense())  # use sparse man

    adj_val = sp.csr_matrix(adj_val)
    adj_val = torch.tensor(adj_val.todense())

    for i in range(adj_train.shape[0]):
        adj_train[i,i] = 1
    for i in range(adj_val.shape[0]):
        adj_val[i,i] = 1

    masked_percent = 15
    adj_train_org = copy.deepcopy(adj_train)
    adj_val_org = copy.deepcopy(adj_val)
    ones = (adj_train == 1).nonzero(as_tuple=False)
    twos = (adj_train == 2).nonzero(as_tuple=False)
    zeros = (adj_train == 0).nonzero(as_tuple=False)

    ones_val = (adj_val == 1).nonzero(as_tuple=False)

    graph_dgl = dgl.graph((ones[:,0], ones[:,1]))
    graph_dgl_val = dgl.graph((ones_val[:,0], ones_val[:,1]))

    masked_1 = random.sample(range(0, len(ones)), int(masked_percent/100*len(ones)))
    masked_0 = random.sample(range(0, len(zeros)), int(masked_percent/100*len(ones)))
    masked_1_random_flip = random.sample(masked_1, int(10/100*len(masked_1)))
    masked_0_random_flip = random.sample(masked_0, int(10/100*len(masked_0)))
    masked_1_random_true = random.sample(masked_1, int(10/100*len(masked_1)))
    masked_0_random_true = random.sample(masked_0, int(10/100*len(masked_0)))
    
    mask = torch.zeros(adj_train.shape[0],adj_train.shape[0])
    mask[ones[masked_1][:,0],ones[masked_1][:,1]]=1
    mask[zeros[masked_0][:,0],zeros[masked_0][:,1]]=1
    twos = torch.cat((twos, ones[masked_1]))
    twos = torch.cat((twos, zeros[masked_0]))
    # adj_train[ones[masked_1][:,0],ones[masked_1][:,1]]=2
    # adj_train[zeros[masked_0][:,0],zeros[masked_0][:,1]]=2
    # adj_train[ones[masked_1_random_flip][:,0],ones[masked_1_random_flip][:,1]]=0
    # adj_train[zeros[masked_0_random_flip][:,0],zeros[masked_0_random_flip][:,1]]=1
    # adj_train[ones[masked_1_random_true][:,0],ones[masked_1_random_true][:,1]]=1
    # adj_train[zeros[masked_0_random_true][:,0],zeros[masked_0_random_true][:,1]]=0
    non_zero = adj_train.nonzero()
    src = non_zero[:,0]
    dst = non_zero[:,1]
    ones = (adj_train == 1).nonzero(as_tuple=False)
    twos = (adj_train == 2).nonzero(as_tuple=False)
    zeros = (adj_train == 0).nonzero(as_tuple=False)
    src_1 = ones[:,0]
    dst_1 = ones[:,1]
    src_2 = twos[:,0]
    dst_2 = twos[:,1]
    dict_edges = {('node', 1, 'node'):(src_1,dst_1), ('node', 2, 'node'):(src_2,dst_2)}
    graph_dgl_masked = dgl.heterograph(dict_edges)
    
    # graph_dgl = dgl.from_scipy(adj_train)

    # origianl_graph_statistics = GS.compute_graph_statistics(np.array(adj_train.todense()) + np.identity(adj_train.shape[0]))
    
    #graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops
    
    #num_nodes = graph_dgl.number_of_dst_nodes()
    # adj_train = torch.tensor(adj_train.todense())  # use sparse man
   
    num_nodes = adj_train.shape[0]
    num_nodes_val = adj_val.shape[0]

       
    if (type(feat_train) == np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
    else:
        feat_train = feat_train
    
    if (type(feat_val) == np.ndarray):
        feat_val = torch.tensor(feat_val, dtype=torch.float32)
    else:
        feat_val = feat_val

    # adj_val = sp.csr_matrix(adj_val)
    # graph_dgl_val = dgl.from_scipy(adj_val)
    # origianl_graph_statistics = GS.compute_graph_statistics(np.array(adj_train.todense()) + np.identity(adj_train.shape[0]))
    # graph_dgl_val.add_edges(graph_dgl_val.nodes(), graph_dgl_val.nodes())  # the library does not add self-loops
    # num_nodes_val = graph_dgl_val.number_of_dst_nodes()
    # adj_val = torch.tensor(adj_val.todense())  # use sparse man
    # if (type(feat_val) == np.ndarray):
    #     feat_val = torch.tensor(feat_val, dtype=torch.float32)
    # else:
    #     feat_val = feat_ 
    # randomly select 25% of the test nodes to not be in evidence
    
    
    not_evidence = random.sample(list(testId), int(0 * len(testId)))

    model = PN_FrameWork(num_of_comunities,
                           encoder=encoder_model,
                           decoder=decoder_model,
                           feature_decoder = feature_decoder,
                           feature_encoder = feature_encoder_model,
                           classifier=class_decoder,
                           not_evidence = not_evidence)  # parameter namimng, it should be dimentionality of distriburion
    

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    pos_wight = torch.true_divide((adj_train_org.shape[0] ** 2 - torch.sum(adj_train_org)), torch.sum(
        adj_train_org))  # addrressing imbalance data problem: ratio between positve to negative instance
    # pos_wight = torch.tensor(1)
    norm = torch.true_divide(adj_train_org.shape[0] * adj_train_org.shape[0],
                             ((adj_train_org.shape[0] * adj_train_org.shape[0] - torch.sum(adj_train_org)) * 2))

    pos_wight_val = torch.true_divide((adj_val_org.shape[0] ** 2 - torch.sum(adj_val_org)), torch.sum(
        adj_val_org))  # addrressing imbalance data problem: ratio between positve to negative instance
    # pos_wight = torch.tensor(1)
    norm_val = torch.true_divide(adj_val_org.shape[0] * adj_val_org.shape[0],
                             ((adj_val_org.shape[0] * adj_val_org.shape[0] - torch.sum(adj_val_org)) * 2))



    not_masked = torch.ones(mask.shape[0], mask.shape[1])-mask

    mask_index = (mask == 1).nonzero(as_tuple=False)
    not_masked_index = (not_masked == 1).nonzero(as_tuple=False)
    
    norm_masked = torch.true_divide(mask_index.shape[0],
                             ((mask_index.shape[0] - torch.sum(adj_train_org*mask)) * 2))

    norm_not_masked = torch.true_divide(not_masked_index.shape[0],
                         ((not_masked_index.shape[0] - torch.sum(adj_train_org*not_masked)) * 2))
 
    pos_wight_masked = torch.true_divide((mask_index.shape[0] - torch.sum(adj_train_org*mask)), torch.sum(
        adj_train_org*mask))
    
    pos_wight_not_masked = torch.true_divide((not_masked_index.shape[0] - torch.sum(adj_train_org*not_masked)), torch.sum(
        adj_train_org*not_masked))

    #pos_weight=neg/pos
    #norm = total/2*neg

    pos_weight_feat = torch.true_divide((feat_train.shape[0]*feat_train.shape[1]-torch.sum(feat_train)),torch.sum(feat_train))
    norm_feat = torch.true_divide((feat_train.shape[0]*feat_train.shape[1]),(2*(feat_train.shape[0]*feat_train.shape[1]-torch.sum(feat_train))))

    pos_weight_feat_val = torch.true_divide((feat_val.shape[0]*feat_val.shape[1]-torch.sum(feat_val)),torch.sum(feat_val))
    norm_feat_val = torch.true_divide((feat_val.shape[0]*feat_val.shape[1]),(2*(feat_val.shape[0]*feat_val.shape[1]-torch.sum(feat_val))))

    # lambda_1 = 0
    # lambda_2 = 1
    # lambda_3 = 1
    # hyperparameter_bounds = [(0, 1), (0, 1)]
    # result = minimize(objective, hyperparameter_bounds, method='L-BFGS-B')



    partial_objective = partial(train_model, labels_train = labels_train, labels_val = labels_val, dataset=dataset, epoch_number=epoch_number, model=model, graph_dgl=graph_dgl, graph_dgl_val=graph_dgl_val, feat_train=feat_train, feat_val=feat_val,  targets=targets, sampling_method=sampling_method, is_prior=is_prior, loss_type=loss_type, adj_train_org=adj_train_org, adj_val_org=adj_val_org, norm_feat=norm_feat, pos_weight_feat=pos_weight_feat, norm_feat_val=norm_feat_val, pos_weight_feat_val=pos_weight_feat_val, num_nodes=num_nodes, num_nodes_val=num_nodes_val, pos_wight=pos_wight, norm=norm, pos_wight_val=pos_wight_val, norm_val=norm_val,optimizer=optimizer )
    # hyperparameter_bounds = {'lambda_1': (0.1, up_bound), 'lambda_2': (0.1, up_bound), 'lambda_3': (0.1, up_bound)}
    # hyperparameter_bounds = {'lambda_1': (0.1, up_bound)}
    # optimizer_hp = BayesianOptimization(f=partial_objective, pbounds=hyperparameter_bounds, allow_duplicate_points=True)
    # optimizer_hp.maximize(init_points=2, n_iter=5, allow_duplicate_points=True)
    # model.load_state_dict(torch.load('best_model_'+dataset+'.pt'))
    # lambda_1 = optimizer_hp.max['params']['lambda_1']
    # lambda_2 = optimizer_hp.max['params']['lambda_2']
    # #lambda_3 = optimizer_hp.max['params']['lambda_3']
    # init_guess = [1, 1]
    # bnds = ((0, 1), (0, 1))
    # res = opt.minimize(partial_objective, init_guess, method='BFGS', bounds=bnds, tol=1e-6, options={'maxiter':5})
    # lambda_1, lambda_2 = res.x

    search_space = [
        Real(0.01, 1, name='lambda_1'),  # Adjust the range as needed
        Real(0.01, 1, name='lambda_2'), # Adjust the range as needed
        Real(0.01, 1, name='lambda_3')  # Adjust the range as needed
    ]
    result = gp_minimize(
        func=partial_objective,
        dimensions=search_space,
        n_calls=10,  # Number of function calls (iterations)
        random_state=10,  # Set a random seed for reproducibility
        n_initial_points=10
    )

    lambda_1, lambda_2, lambda_3 = result.x

    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train, targets, sampling_method,
                                                                     is_prior, train=True)
        # compute loss and accuracy
        z_kl, reconstruction_loss, acc, val_recons_loss, loss_adj, loss_feat = optimizer_VAE_pn(lambda_1, lambda_2,lambda_3, labels_train, re_labels, loss_type,
                                                                           reconstructed_adj,
                                                                           reconstructed_feat,
                                                                           adj_train_org, feat_train, norm_feat,
                                                                           pos_weight_feat,
                                                                           std_z, m_z, num_nodes, pos_wight, norm)
        loss = reconstruction_loss + z_kl

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        # print some metrics
        print(
            "Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
                 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
        print("lambdas:", result.x)
    model.eval()

    return model, z

def train_model(params, labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train, feat_val,  targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat, pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm, pos_wight_val, norm_val,optimizer ):
    lambda_1, lambda_2, lambda_3 = params
    best_auc = 0
    with open('./results_csv/best_auc.csv', newline='') as f:
        reader = csv.DictReader(f)
        for q in reader:
            best_auc = float(q['auc'])

    # best_validation_loss = 0

    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train, targets, sampling_method,
                                                                     is_prior, train=True)
        # compute loss and accuracy
        z_kl, reconstruction_loss, acc, val_recons_loss, loss_adj, loss_feat = optimizer_VAE_pn(lambda_1,lambda_2,lambda_3, labels_train, re_labels, loss_type, reconstructed_adj,
                                                                           reconstructed_feat,
                                                                           adj_train_org, feat_train, norm_feat,
                                                                           pos_weight_feat,
                                                                           std_z, m_z, num_nodes, pos_wight, norm)
        loss = reconstruction_loss + z_kl

        # reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()

        model.eval()

        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some metrics
        print(
            "Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
                epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
    model.eval()
    with torch.no_grad():

        std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(graph_dgl_val, feat_val, labels_val, targets,  sampling_method,
                                                                 is_prior, train=True)
        z_kl_val, val_reconstruction_loss, val_acc, val_recons_loss, loss_adj, loss_feat = optimizer_VAE_pn(lambda_1,lambda_2,lambda_3, labels_val, re_labels_val, loss_type, reconstructed_adj_val,
                                                                       reconstructed_feat_val,
                                                                       adj_val_org, feat_val, norm_feat_val,
                                                                       pos_weight_feat_val,
                                                                       std_z_val, m_z_val, num_nodes_val, pos_wight_val, norm_val)
        val_loss_total = val_reconstruction_loss+z_kl_val
        # with open('./results_csv/loss_val.csv', 'a') as f:
        #     wtr = csv.writer(f)
        #     wtr.writerow([loss.item()])

    # y_true_feat = (torch.flatten(feat_val)).cpu().detach().numpy()
    # y_pred_feat = (torch.flatten(torch.sigmoid(reconstructed_feat_val))).cpu().detach().numpy()
    # index_sample_0_feat = np.random.choice(np.where(y_true_feat == 1)[0], 100)
    # index_sample_1_feat = np.random.choice(np.where(y_true_feat == 0)[0], 100)
    # index_sample_feat = np.concatenate((index_sample_0_feat, index_sample_1_feat))
    # auc_feat = roc_auc_score(y_score=y_pred_feat[index_sample_feat], y_true=y_true_feat[index_sample_feat])

    # y_true_adj = (torch.flatten(adj_val_org)).cpu().detach().numpy()
    # y_pred_adj = (torch.flatten(torch.sigmoid(reconstructed_adj_val))).cpu().detach().numpy()
    # index_sample_0_adj = np.random.choice(np.where(y_true_adj == 1)[0], 100)
    # index_sample_1_adj = np.random.choice(np.where(y_true_adj == 0)[0], 100)
    # index_sample_adj = np.concatenate((index_sample_0_adj, index_sample_1_adj))
    # auc_adj = roc_auc_score(y_score=y_pred_adj[index_sample_adj], y_true=y_true_adj[index_sample_adj])
    #
    # auc_labels = roc_auc_score(y_score= re_labels_val, y_true= labels_val)
    #
    # auc_val = -1*(auc_feat+auc_adj+auc_labels)
    #
    # if best_auc > auc_val:
    #     best_auc = auc_val
    #     torch.save(model.state_dict(), 'best_model_' + dataset + '.pt')
    #     with open('./results_csv/best_auc.csv', 'a') as f:
    #         wtr = csv.writer(f)
    #         wtr.writerow([best_auc])
    #
    #
    #
    # return auc_val

    w_l = weight_labels(labels_val)
    posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(reconstructed_adj_val, adj_val_org, pos_weight=pos_wight_val)
    posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat_val , feat_val,
                                                                             pos_weight=pos_weight_feat)
    posterior_cost_classes =  F.cross_entropy(re_labels_val,(torch.tensor(labels_val).to(torch.float64)),weight=w_l)

    cost = posterior_cost_edges+posterior_cost_features+posterior_cost_classes

    return cost.item()








# z_kl, reconstruction_loss, acc, val_recons_loss = optimizer_VAE_em(alpha, mask_index, not_masked_index, reconstructed_adj, reconstructed_feat,
#                                                                adj_train_org, feat_train, norm_feat,pos_weight_feat,
#                                                                std_z, m_z, num_nodes, pos_wight_masked, pos_wight_not_masked, norm_masked, norm_not_masked )


# train_auc, train_acc, train_ap, train_conf = roc_auc_estimator_train(train_true, train_false,
#                                                       reconstructed_adj, adj_train)

# if split_the_data_to_train_test == True:
#     std_z, m_z, z, reconstructed_adj_val = model(graph_dgl_val, feat_val, is_prior, train=False)
#     reconstructed_adj_val = torch.sigmoid(reconstructed_adj_val).detach().numpy()
#     val_auc, val_acc, val_ap, val_conf = roc_auc_estimator_train(val_true, val_false,
#                                                     reconstructed_adj_val, adj_val)

#     # keep the history to plot
#     pltr.add_values(epoch, [loss.item(), train_acc,  reconstruction_loss.item(), z_kl, train_auc],
#                     [None, val_acc, val_recons_loss,None, val_auc  # , val_ap
#                         ], redraw=False)  # ["Accuracy", "Loss", "AUC", "AP"]
# else:
#     # keep the history to plot
#     pltr.add_values(epoch, [acc, loss.item(), None  # , None
#                             ],
#                     [None, None, None  # , None
#                       ], redraw=False)  # ["Accuracy", "loss", "AUC", "AP"])

# # Ploting the recinstructed Graph
# if epoch % visulizer_step == 0:
#     # pltr.redraw()
#     print("Val conf:", )
#     print(val_conf, )
#     print("Train Conf:")
#     print(train_conf)