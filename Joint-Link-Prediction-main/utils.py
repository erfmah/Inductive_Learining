import sys
import os
import torch
import random
import math
import csv
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, recall_score, \
    precision_score, precision_recall_curve
from numpy import argmax
import copy
import scipy.sparse as sp
import numpy as np
from scipy import sparse
import dgl
from scipy.stats import multivariate_normal
import torch.nn.functional as F



# objective Function
def optimizer_VAE(pred, labels, std_z, mean_z, num_nodes, pos_wight, norm):
    val_poterior_cost = 0
    posterior_cost = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight)

    z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))

    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    return z_kl, posterior_cost, acc, val_poterior_cost


def get_metrics(target_edges, org_adj, reconstructed_adj):
    reconstructed_adj =  sparse.csr_matrix(torch.sigmoid(reconstructed_adj).detach().numpy())
    org_adj = sparse.csr_matrix(org_adj)
    prediction = []
    true_label = []
    counter = 0
    for edge in target_edges:
        prediction.append(reconstructed_adj[edge[0], edge[1]])
        prediction.append(reconstructed_adj[edge[1], edge[0]])
        true_label.append(org_adj[edge[0], edge[1]])
        true_label.append(org_adj[edge[1], edge[0]])

    pred = np.array(prediction)
    
    
    precision, recall, thresholds = precision_recall_curve(true_label, pred)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = argmax(fscore)
    Threshold = thresholds[ix]
    Threshold = 0.5
    
    
    pred[pred > Threshold] = 1.0
    pred[pred < Threshold] = 0.0
    pred = pred.astype(int)


    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)

    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:] # dividing by 5 to get top 20%
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
    
    
    return auc, acc, ap, precision, recall, HR



def roc_auc_single(prediction, true_label):
    pred = np.array(prediction)
    pred[pred > .5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)

    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)
    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:] # dividing by 5 to get top 20%
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])  
    pred = np.array(prediction)
    
    return auc, acc, ap, precision, recall, HR





def run_network(feats, adj, model, targets, sampling_method, is_prior):
    adj = sparse.csr_matrix(adj)
    graph_dgl = dgl.from_scipy(adj)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
    std_z, m_z, z, re_adj = model(graph_dgl, feats, targets, sampling_method, is_prior, train=False)
    return std_z, m_z, z, re_adj


def get_pdf(mean_p, std_p, mean_q, std_q, z, targets):

    pdf_all_z_p = 0
    pdf_all_z_q = 0
    for i in targets:
        # TORCH
        cov_p = np.diag(std_p.detach().numpy()[i] ** 2)
        dist_p = torch.distributions.multivariate_normal.MultivariateNormal(mean_p[i], torch.tensor(cov_p))
        pdf_all_z_p += dist_p.log_prob(z[i]).detach().numpy()

        cov_q = np.diag(std_q.detach().numpy()[i] ** 2)
        dist_q = torch.distributions.multivariate_normal.MultivariateNormal(mean_q[i], torch.tensor(cov_q))
        pdf_all_z_q += dist_q.log_prob(z[i]).detach().numpy()
    return pdf_all_z_p, pdf_all_z_q













