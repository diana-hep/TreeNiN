
[![DOI](https://zenodo.org/badge/160135404.svg)](https://zenodo.org/badge/latestdoi/160135404)

# Recursive Neural Network for Jet Physics

**Sebastian Macaluso and Kyle Cranmer**


Note that this is an early development version. 

## Introduction

In this method, a recursive neural network (RecNN) is trained on jet trees. The RecNN provides a ``jet embedding", which maps a set of 4-momenta into a vector of fixed size and can be trained together with a successive network used for classification or regression  (see [Louppe et. al. 2017, "QCD-Aware Recursive Neural Networks for Jet Physics"](https://arxiv.org/abs/1702.00748) for more details). Jet constituents are reclustered to form binary trees, and the topology is determined by the clustering algorithm (e.g. kt, anti-kt or Cambridge/Aachen). We chose the kt clustering algorithm, and 7 features for the nodes: |p|, eta, phi, E, E/Ejet, pT and theta. 
We removed the median and scaled each feature according to the range between the first and the third quartiles (this scaling is robust to outliers).

To make training fast enough, a special batching was implemented in [Louppe et. al. 2017]. Jets are reorganized by levels (e.g. the root node of each tree in the batch is added at level zero, their children at level one, etc). Each level is restructured so that all internal nodes come first, followed by outer nodes (leaves), and zero padding is applied when necessary.

Results are obtained from a PyTorch implementation to do GPU batch training of a Network in Network RecNN (NiN RecNN). The NiN RecNN is a modification of the simple RecNN architecture introduced in [Louppe et. al. 2017], where we add fully connected layers at each node of the binary tree before moving forward to the next level. In particular, we added 2 NiN layers with ReLU activations. Also, we split weights between internal nodes and leaves, both for the NiN layers and for the convolution of the 7 input features of each node. Finally, we introduce two sets of independent weights for the NiN layers of the left and right children of the root node. Our model has 62,651 trainable parameters. Training is performed over 40 epochs with a minibatch of 128 and a learning rate of 0.002 (decayed by a factor of 0.9 after every epoch), using the cross entropy loss function and Adam as the optimizer. 


## Getting started

A description and link to a full dataset (provided by Gregor Kasieczka, Michael Russel and Tilman Plehn) can be found [here](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit)
with the link to download it [here](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6). This dataset contains 1.2M training events, 400k validation events, 400k test events with equal numbers of top quark and qcd jets.



1. Create the directories `in_data` and `out_data` in [top_reference_dataset](top_reference_dataset) and upload the dataset to `in_data` 

### Package structure

`data` dir  
 
- inputTress: All the raw data with the jet clustering history
- input_batches_pad: Batches of input data for the RecNN


=================================================================================================
RecNN
Working dir with the batch training implementation 

-------------------------------------------------------------------------
search_hyperparams.py: main file that calls train.py and evaluate.py, and runs hyperparameters searches

Parameters to specify before running:
sample_name (this specifies the filenames to be used from data/inputTrees)
multi_scan function arguments

To run:
python search_hyperparams.py --gpu=0 &> experiments/simple_rnn_kt/log_gpu0 &

-------------------------------------------------------------------------
train.py: Call all the other files to load the raw data, create the jet trees, create the batches and train the model. This file calls model/recNet.py, model/data_loader.py, model/preprocess.py and utils.py

Parameters to specify before running:
make_batch=True/False (To create the batched dataset from the raw data, or use a previously generated batched dataset)
pT_order=True/False (To reorganize the trees based on pT)
nyu=True/False (To run over the datasets of arXiv:1702.00748. This datasets are not provided here)
sample_name (if running from search_hyperparams.py, then “sample_name” will be overwritten)
sg (signal label)
bg (background label)

To run as a stand-alone code:
CUDA_VISIBLE_DEVICES=1 python -u train.py --model_dir=experiments/nyu_antikt-kt-delphes_test &> experiments/nyu_antikt-kt-delphes_test/log_nyu_antikt-kt-delphes &

-------------------------------------------------------------------------
evaluate.py: loads the weights that give the best val accuracy and gets the accuracy, tpr, fpr, and ROC auc over the test set.

Parameters to specify before running:
nyu
sample_name (if running from search_hyperparams.py, then “sample_name” will be overwritten)

To run as a stand-alone code:
CUDA_VISIBLE_DEVICES=2 python evaluate.py --model_dir=experiments/nyu_antikt-antikt > experiments/nyu_antikt-antikt/log_eval_nyu_antikt-antikt

-------------------------------------------------------------------------
model/recNet.py: model architecture for batch training and accuracy function.

model/data_loader.py: load the raw data and create the batches:

 - Load the jet events and make the trees.
 - Split the sample into train, cross-validation and test with equal number of sg and bg events. Then shuffle each set.
 - Load the jet trees, reorganize the tree by levels, create a batch of N jets by appending the nodes of each jet to each level and add zero padding so that all the levels have the same size
 - Generator function that loads the batches, shifts numpy arrays to torch tensors and feeds the training/validation pipeline
 
 
model/preprocess.py: rewrite and reorganize the jet contents (e.g. add features for each node such as energy, pT, eta, phi, charge, muon ID, etc) 

-------------------------------------------------------------------------
experiments/template_params.json:  Template file that contains all the architecture parameters and training hyperparameters for a specific run. “search_hyperparams.py” modifies these parameters for each scan

experiments/dir_name: dir with all the hyperparameter scan results (weights, log files, results) for each sample/architecture

-------------------------------------------------------------------------
utils.py: auxiliary functions for training, logging, loading hyperparameters from json file, etc.

-------------------------------------------------------------------------
synthesize_results.py: Aggregate metrics of all experiments and outputs a file located at “experiments/dir_name” with a  list where they are sorted. Currently sorts based on best ROC auc.

To run: python synthesize_results.py

## Acknowledgements

SM gratefully acknowledges the support of NVIDIA Corporation with the donation of a Titan V GPU used for this research.

## References

If you use this package, please cite this code as

```
@misc{RecNN,
author       = "Macaluso, Sebastian and Cranmer, Kyle",
title        = "{Recursive neural network for jet physics}",
note         = "{DOI: 10.5281/zenodo.2582216}",
year         = {2019},
url          = {https://github.com/SebastianMacaluso/RecNN_PyTorch}
}
```



















