
[![DOI](https://zenodo.org/badge/160135404.svg)](https://zenodo.org/badge/latestdoi/160135404)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Tree Network in Network (TreeNiN) for Jet Physics

**Sebastian Macaluso and Kyle Cranmer**


Note that this is an early development version. 

## Introduction

In this method, a tree neural network (TreeNN) is trained on jet trees. The TreeNN provides a ``jet embedding", which maps a set of 4-momenta into a vector of fixed size and can be trained together with a successive network used for classification or regression  (see [Louppe et. al. 2017, "QCD-Aware Recursive Neural Networks for Jet Physics"](https://arxiv.org/abs/1702.00748) for more details). Jet constituents are reclustered to form binary trees, and the topology is determined by the clustering algorithm (e.g. kt, anti-kt or Cambridge/Aachen). We chose the kt clustering algorithm, and 7 features for the nodes: |p|, eta, phi, E, E/Ejet, pT and theta. We scaled each feature with the scikit-learn preprocessing method RobustScaler (this scaling is robust to outliers).

To speed up the training, a special batching was implemented in [Louppe et. al. 2017]. Jets are reorganized by levels (e.g. the root node of each tree in the batch is added at level zero, their children at level one, etc). Each level is restructured so that all internal nodes come first, followed by outer nodes (leaves), and zero padding is applied when necessary. Results are obtained from a PyTorch implementation to provide GPU acceleration.

We introduced a Network in Network generalization of the simple TreeNN architecture introduced in [Louppe et. al. 2017], where we add fully connected layers at each node of the binary tree before moving forward to the next level. We refer to this model as TreeNiN. In particular, we add 2 NiN layers with ReLU activations. Also, we split weights between internal nodes and leaves, both for the NiN layers and for the initial embedding of the 7 input features of each node. Finally, we introduce two sets of independent weights for the NiN layers of the left and right children of the root node. Our model has 62,651 trainable parameters. Training is performed over 40 epochs with a minibatch of 128 and a learning rate of 0.002 (decayed by a factor of 0.9 after every epoch), using the cross entropy loss function and Adam as the optimizer. 


## Dependencies

Make sure the following tools are installed and running:

- Python 2.7, Python 3.6, packages included in [Anaconda](https://www.anaconda.com/) (Numpy, Scipy, scikit-learn, Pandas), [PyROOT](https://root.cern.ch/pyroot), [FastJet](http://fastjet.fr/), [PyTorch](https://pytorch.org/).

### Using the Tree Neural Network for the Top Tag Reference Dataset

A description and link to the Top Tag Reference Dataset (provided by Gregor Kasieczka, Michael Russel and Tilman Plehn) can be found [here](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit)
with the link to download it [here](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6). This dataset contains 1.2M training events, 400k validation events, 400k test events with equal numbers of top quark and qcd jets. Only 4 momentum vectors of the jet constituents.

## Data Pipeline

### Package structure in [`top_reference_dataset`](top_reference_dataset):

-`Read_data.ipynb`: Loads the datasets in h5 format, gets the labels and non-zero values for the jet constituents and save pickle files in `out_data`.

-`toptag_reference_dataset_Tree.py`: Loads and reclusters the jet constituents. Creates binary trees with the clustering history of the jets and outputs a dictionary for each jet that contains the root_id, tree, content (constituents 4-momentum vectors), mass, pT, energy, eta, phi (also charge, muon ID, etc depending on the information contained in the dataset). Auxiliary scripts that are called:

   - `analysis_functions.py`: auxiliary functions. 
   - `preprocess_functions.py`: auxiliary functions. 
   - `tree_cluster_hist.py`: Creates binary trees. If the dataset has more info besides the jet constituents 4-vectors (e.g. charge, muon ID, etc), it runs a recursive function that traverses the tree down to the leaves and then goes back to the root generating the inner nodes values by adding the children values for the extra info.
   - `jet_image_trim_pt800-900_card.dat`: parameters card with kinematics variables values, clustering algorithm choice, etc. Currently not used.


### Running the data pipeline

1.  Go to [`top_reference_dataset`](top_reference_dataset) dir. Upload the dataset to [`in_data`](top_reference_dataset/in_data) 
2. Run the `h5_to_npy` function in `Read_data.ipynb` to load the datasets in h5 format, get the labels and non-zero values for the jet constituents and save pickle files in `out_data`.

3. Run:

    - *python2.7 toptag_reference_dataset_Tree.py jet_image_trim_pt800-900_card.dat val top_reference_dataset/out_data/ ../data/inputTrees/top_tag_reference_dataset/*

        (The output file with the dictionary for each jet will be saved in [`data/inputTrees/top_tag_reference_dataset/`](data/inputTrees/top_tag_reference_dataset/). Also, change val for train or test to get the run the data pipeline over the test and train sets as well.)



-------------------------------------------------------------------------
## TreeNiN

### Package structure in [`recnn`](recnn):

 - `search_hyperparams.py`: main script that calls preprocess_main.py, train.py and evaluate.py, and runs hyperparameters searches.
    - Parameters to specify before running:
        - multi_scan function arguments, e.g.:
            ```
            multi_scan(learning_rates=[5e-4],
                       decays=[0.92], 
                       batch_sizes=[64], 
                       num_epochs=[40], 
                       hidden_dims=[40], 
                       jet_numbers=[1200000], 
                       Nfeatures=7, 
                       dir_name='top_tag_reference_dataset', 
                       name=architecture+'_kt_2L4WleavesInnerNiNuk', 
                       info='',
                       sample_name=args.sample_name,
                       Nrun_start=0,
                       Nrun_finish=9) 
            ```
        - Flags: *PREPROCESS*, *TRAIN_and_EVALUATE*, *EVALUATE*
 
    - To run:
        *python search_hyperparams.py --gpu=0*
     
 - `preprocess_main.py`: Preprocess the jet dictionaries and go from the jet constituents 4-momentum vectors to the 7 features for the nodes (|p|, eta, phi, E, E/Ejet, pT and theta).
 
 
 - `train.py`: Call all the other files to load the raw data, create the jet trees, create the batches and train the model. This file calls model/recNet.py, model/data_loader.py, model/preprocess.py and utils.py
 
- `evaluate.py`: loads the weights that give the best val accuracy and gets the accuracy, tpr, fpr, and ROC auc over the test set.
 
 - [`model`](model/):
 
    - `recNet.py`: model architecture for batch training and accuracy function.
 
    - `data_loader.py`: load the raw data and create the batches:
    
        - Load the jet events and make the trees.
        - Split the sample into train, cross-validation and test with equal number of sg and bg events. Then shuffle each set.
        - Load the jet trees, reorganize the tree by levels, create a batch of N jets by appending the nodes of each jet to each level and add zero padding so that all the levels have the same size
        - Generator function that loads the batches, shifts numpy arrays to torch tensors and feeds the training/validation pipeline
 
 
    - `preprocess.py`: rewrite and reorganize the jet contents (e.g. add features for each node such as energy, pT, eta, phi, charge, muon ID, etc) 
 
    - `dataset.py`: This script defines:
        - A new subclass of torch.utils.data.Dataset that overrides the *__len__*  and *__getitem__* methods. This allows to call torch.utils.data.DataLoader to generate the batches in parallel with num_workers>1 and speed up the code. 
        - A customized *collate* function to load the batches with torch.utils.data.DataLoader.
 
-[`experiments`](experiments):
 
   - `template_params.json`:  Template file that contains all the architecture parameters and training hyperparameters for a specific run. “search_hyperparams.py” modifies these parameters for each scan
 
 - `experiments/[dir_name]`: dir with all the hyperparameter scan results (weights, log files, results) for each sample/architecture. For each run, a new directory will be created with files saving the output probabilities on the test set, metrics history after each epoch, best and last weights, etc.
 
 -`jet_study.ipynb`: Load results from `experiments/[dir_name]`, and get statistics for single and/or multiple runs.

-`utils.py`: auxiliary functions for training, logging, loading hyperparameters from json file, etc.
 
 -------------------------------------------------------------------------
 ### Running the TreeNiN 
 
 1. Set the flag *PREPROCESS=True* and the other ones to False in  `search_hyperparams.py` (This will run `preprocess_main.py` and save the preprocessed in [`../data/preprocessed_trees/`](../data/preprocessed_trees/)). Run `search_hyperparams.py`.
 
 3. Set the flag *TRAIN_and_EVALUATE=True* and the other ones to False in  `search_hyperparams.py` (This will run `train.py` and `evaluate.py`). Specify the *multi_scan function* arguments and run `search_hyperparams.py`. Results will be saved in `experiments/[dir_name]`.


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



















