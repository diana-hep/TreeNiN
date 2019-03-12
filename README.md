
[![DOI](https://zenodo.org/badge/160135404.svg)](https://zenodo.org/badge/latestdoi/160135404)

# Tree Network in Network (TreeNiN) for Jet Physics

**Sebastian Macaluso and Kyle Cranmer**


Note that this is an early development version. 

## Introduction

In this method, a tree neural network (TreeNN) is trained on jet trees. The TreeNN provides a ``jet embedding", which maps a set of 4-momenta into a vector of fixed size and can be trained together with a successive network used for classification or regression  (see [Louppe et. al. 2017, "QCD-Aware Recursive Neural Networks for Jet Physics"](https://arxiv.org/abs/1702.00748) for more details). Jet constituents are reclustered to form binary trees, and the topology is determined by the clustering algorithm (e.g. kt, anti-kt or Cambridge/Aachen). We chose the kt clustering algorithm, and 7 features for the nodes: |p|, eta, phi, E, E/Ejet, pT and theta. We scaled each feature with the scikit-learn preprocessing method RobustScaler (this scaling is robust to outliers).

To speed up the training, a special batching was implemented in [Louppe et. al. 2017]. Jets are reorganized by levels (e.g. the root node of each tree in the batch is added at level zero, their children at level one, etc). Each level is restructured so that all internal nodes come first, followed by outer nodes (leaves), and zero padding is applied when necessary. Results are obtained from a PyTorch implementation to provide GPU acceleration.

We introduced a Network in Network generalization of the simple TreeNN architecture introduced in [Louppe et. al. 2017], where we add fully connected layers at each node of the binary tree before moving forward to the next level. We refer to this model as TreeNiN. In particular, we add 2 NiN layers with ReLU activations. Also, we split weights between internal nodes and leaves, both for the NiN layers and for the initial embedding of the 7 input features of each node. Finally, we introduce two sets of independent weights for the NiN layers of the left and right children of the root node. Our model has 62,651 trainable parameters. Training is performed over 40 epochs with a minibatch of 128 and a learning rate of 0.002 (decayed by a factor of 0.9 after every epoch), using the cross entropy loss function and Adam as the optimizer. 


## Getting started

### Dependencies

Make sure the following tools are installed and running:

- Python 2.7, Python 3.6, packages included in [Anaconda](https://www.anaconda.com/) (Numpy, Scipy, scikit-learn, Pandas), [PyROOT](https://root.cern.ch/pyroot), [FastJet](http://fastjet.fr/), [PyTorch](https://pytorch.org/).

### Using the Tree Neural Network

A description and link to a full dataset (provided by Gregor Kasieczka, Michael Russel and Tilman Plehn) can be found [here](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit)
with the link to download it [here](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6). This dataset contains 1.2M training events, 400k validation events, 400k test events with equal numbers of top quark and qcd jets. Only 4 momentum vectors of the jet constituents.



1.  Go to [`top_reference_dataset`](top_reference_dataset) dir. Upload the dataset to [`in_data`](top_reference_dataset/in_data) 
2. `Read_data.ipynb`: Run the `h5_to_npy` function to load the datasets in h5 format, get the labels and non-zero values for the jet constituents and save pickle files in `out_data`.
3. `toptag_reference_dataset_Tree.py`: Loads the jet constituents, reclusters the jet constituents and creates binary trees with the clustering history of the jets and outputs a dictionary for each jet that contains the root_id, tree, content (constituents 4-momentum vectors), mass, pT, energy, eta, phi (also charge, muon ID, etc depending on the information contained in the dataset). Auxiliary scripts that are called:
    
    - `analysis_functions.py`
    - `preprocess_functions.py`
    - `tree_cluster_hist.py`
    - `jet_image_trim_pt800-900_card.dat`: param_card with kinematics variables values, clustering algorithm, etc. Currently not used.

- jet_preprocessTree.py:  This is the main script that loads .root files in a dir and outputs the tree lists ready to be loaded by the second part of the pipeline. The parameters are specified in “jet_image_trim_pt800-900_card.dat”. This script calls auxiliary ones: “analysis_functions.py”, “tree_cluster_hist.py” and “preprocess_functions.py”. (Run with "python2.7" and not "python" on the hexfarm, if not there are issues with matplotlib)

. Reclusters the Delphes jets and applies trimming (it’s probably straightforward to add pruning as well)
. Applies match and merge requirements
. Applies preprocessing: shift-rotate-flip. I had to modify the flip functions to do it before pixelating. 
. Added charge, abs(charge) and muon id as labels to each FastJet Pseudojet and reclusters the jet.
. Runs a recursive function to access the tree clustering history and create the trees, also adding charge, abs(charge) and muon id lists (only for the leaves).
. Runs a recursive function that traverses the tree down to the leaves and then goes back to the root generating the inner nodes values by adding the children values for charge, abs(charge) and muon id






python2.7 toptag_reference_dataset_Tree.py jet_image_trim_pt800-900_card.dat val top_tag_reference_dataset/in_jets/ top_tag_reference_dataset/tree_list

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



















