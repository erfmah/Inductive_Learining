# Joint Link Prediction Via Inference from a Model
**Authors**: Parmis Naddaf, Erfaneh Mahmoudzadeh Ahmadi Nejad, Kiarash Zahirnia, Manfred Jaeger, Oliver Schulte

This is a PyTorch implementation of Joint Link Prediction Via Inference from a Model.
## Overview
A Joint Link Prediction Query (JLPQ) asks for the probability of a target **set of links**, given an evidence set of links and node attributes. Supporting inference to answer JLPQs is a new use case for a Graph Generative Model (GGM). Such a query answering system facilitates applying graph prediction in a production environment where multiple users pose a range of queries to be answered. In this paper we showed inference from a trained Variational Graph Auto-Encoder (VGAE) model can be used to answer JLPQs, in zero-shot manner without retraining the model. The key issue is how to apply a graph encoder when some links are unspecified by the query. For more information you can see our paper ["Joint Link Prediction Via Inference from a Model"](www.google.com).

## Run
To execute the main.py file, run the model, and address the joint link prediction queries, you must customize the parameters within the command.
For example this command runs the code in Fully inductive setting, with Monte Carlo sampling, GAT encoder for single link prediction on Cora dataset:
```sh
python main.py --fully_inductive True --encoder_type "Multi_GAT" --sampling_method "monte" --method "single" --dataSet "Cora" 
```


## Semi/Fully Inductive Setting

In fully inductive setting, all query nodes are test nodes. The evidence set comprises all links between test nodes that are not target links. You can run the model fully inductive by assigning any value to "--fully_inductive" parameter:
```sh
python main.py --fully_inductive True
```
Or you can run the model in semi inductive setting where, Each target link connects at least one test node. The evidence set comprises all links from the input graph that are not target links. By default, the model operates in the semi-inductive setting.
## Encoder Types
You can run this model with three different encoders by using following commands:
- [**VGAE-GCN**](https://openreview.net/pdf?id=SJU4ayYgl) Graph Convolutional Neural Network is a popular encoder:
    - ```python main.py --encoder_type "Multi_GCN"```
- [**VGAE-GAT**](https://openreview.net/forum?id=rJXMpikCZ) Graph Attention Networks add link attention weights to graph convolutions:
    - ```python main.py --encoder_type "Multi_GAT" ```
- [**VGAE-GIN**](https://openreview.net/pdf?id=ryGs6iA5Km) The Graph Isomorphism Network is a type of GNN that consists of two steps of         aggregation and combination:
    - ```python main.py --encoder_type "Multi_GIN" ```


## Sampling Methods
During the inference step, you can choose from three available sampling methods. You can select the desired sampling method using the following commands:
- **Deterministic inference**:
    - ```python main.py --sampling_method "deterministic" ```
- **Monte Carlo inference**:
    - ```python main.py --sampling_method "monte" ```
- **Importance Sampling inference** : 
    - ```python main.py --sampling_method "importance_sampling" ```

## Single/Multi Link Prediction
This model can answer two types of queries:
- **Single Link Queries**, where each query has one target edge. This is the traditional link prediction setup.
    - ```python main.py --method "single" ```
- **Joint Link Prediction Queries**, where each query has at least one target edge. This is the novel task considered in our paper.
    - ```python main.py --method "multi" ```


## Data
You have the option to select a dataset from a set of available options such as Cora and CiteSeer datasets. To use a specific dataset, you can execute the following command:
```sh
python main.py --dataSet "CiteSeer"
```
Although we've provided the CiteSeer dataset due to space constraints, you can access all the datasets through [this link](www.google.com). 

## Environment
- python=3.8
- scikit-learn==1.2.2
- scipy==1.10.1
- torchmetrics==0.11.4
- python-igraph==0.10.4
- powerlaw=1.4.6
- dgl==1.2a230610+cu117
- dglgo==0.0.2



## Cite
If you find the code useful, please cite our papers.
```sh
cite
```
