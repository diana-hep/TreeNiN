# Tree Network in Network (TreeNiN) for Jet Physics

### **Sebastian Macaluso and Kyle Cranmer**

Note that this is an early development version. 

[![DOI](https://zenodo.org/badge/160135404.svg)](https://zenodo.org/badge/latestdoi/160135404) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

In this method, a tree neural network (TreeNN) is trained on jet trees. The TreeNN provides a *jet embedding*, which maps a set of 4-momenta into a vector of fixed size and can be trained together with a successive network used for classification or regression  (see [Louppe et al. 2017, "QCD-Aware Recursive Neural Networks for Jet Physics"](https://arxiv.org/abs/1702.00748) for more details). Jet constituents are reclustered to form binary trees, and the topology is determined by the clustering algorithm (e.g. kt, anti-kt or Cambridge/Aachen). We chose the kt clustering algorithm, and 7 features for the nodes: |p|, eta, phi, E, E/Ejet, pT and theta. We scaled each feature with the scikit-learn preprocessing method RobustScaler (this scaling is robust to outliers).

To speed up the training, a special batching was implemented in [Louppe et al. 2017](https://arxiv.org/abs/1702.00748). Jets are reorganized by levels (e.g. the root node of each tree in the batch is added at level zero, their children at level one, etc). Each level is restructured so that all internal nodes come first, followed by outer nodes (leaves), and zero padding is applied when necessary. We developed a PyTorch implementation to provide GPU acceleration.

We introduced a Network in Network generalization of the simple TreeNN architecture proposed in [Louppe et al. 2017](https://arxiv.org/abs/1702.00748), where we add fully connected layers at each node of the binary tree before moving forward to the next level. We refer to this model as TreeNiN. In particular, we add 2 NiN layers with ReLU activations. Also, we split weights between internal nodes and leaves, both for the NiN layers and for the initial embedding of the 7 input features of each node. Finally, we introduce two sets of independent weights for the NiN layers of the left and right children of the root node. Our model has 33,901 trainable parameters. Training is performed over 40 epochs with a minibatch of 128 and a learning rate of 0.002 (decayed by a factor of 0.9 after every epoch), using the cross entropy loss function and Adam as the optimizer. 


-------------------------------------------------------------------------
## Implementing the TreeNiN on the *Top Tagging Reference Dataset*

This repository includes all the code needed to implement the TreeNiN on the *Top Tagging Reference Dataset*. A description and link to the Top Tagging Reference Dataset (provided by Gregor Kasieczka, Michael Russel and Tilman Plehn) can be found [here](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit)
with the link to download it [here](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6). This dataset contains 1.2M training events, 400k validation events, 400k test events with equal numbers of top quark and qcd jets. Only 4 momentum vectors of the jet constituents.

-------------------------------------------------------------------------
## Running with Docker (Recommended)

### Dependencies:

Install [docker](https://docs.docker.com/install/) 

### Docker Image

1. If building the Docker Image, from the root directory of this repository, run: `docker build --tag=treenin:1.0.0 .`

2. Alternatively, it is easier to download the pre-built image from Docker Hub [here](https://cloud.docker.com/repository/docker/smacaluso/treenin/general):

`docker pull smacaluso/treenin:1.0.0`

**Relevant Structure**:

- [`Dockerfile`](Dockerfile)  
- [`scripts`](scripts): dir with the scripts to install specific dependencies when building the image.  
- [`code`](code): working directory for the docker container.    
    - [`top_reference_dataset`](code/top_reference_dataset)  
        - [`outProb`](code/top_reference_dataset/outProb): dir with the output probabilities.    
        - [`in_data`](code/top_reference_dataset/in_data): dir where the initial test dataset will be downloaded.  
    - [`dataWorkflow.py`](code): script with the data workflow.  
    - [`MLWorkflow.py`](code): script with the machine learning workflow.  
    - [`saveProb.py`](code): script that saves the output probabilities in [`outProb/[filename.pkl]`](code/top_reference_dataset/outProb).  
    - [`recnn`](code/recnn): dir with the code for the TreeNiN.  
    - [`data`](code/data): dir with the jet trees (before and after preprocessing).  


### Running the treenin container: 

Run the container:  
`docker run [options] -i -t smacaluso/treenin:1.0.0 /bin/bash` 

If loading external files, we should mount the host file inside the docker container at running time. This will import a copy of the file to the chosen location (it also creates the path inside the container if neccesary), e.g. for macOS:  
`docker run -v [path/to/external/file]:[full/path/to/file/inside/the/container] [options] -i -t smacaluso/treenin:1.0.0 /bin/bash` 

Check `docker run --help` for all the options. Some useful `[options]`:
- `--rm`: Remove the container when we exit.
- `--shm-size [RAM size]G`: shared memory with the container.

#### Evaluation mode

The container is ready to run the code on evaluation mode (to run on training mode, follow the instructions on *Training mode* below) . From the [working directory](/code/):

1. Choose one of the two options below to run the data workflow: 

    a. Download the [test dataset](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6) from the website:
`python dataWorkflow.py 0`
            
    b. Load the [test dataset](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6) as an external file (first mount the host file inside the docker container at running time):
                `python dataWorkflow.py 0 [full/path/to/file/inside/the/container]`
    
    If running with `0` as an input argument, the whole dataset will be used (this value determines the starting jet position to load from the dataset). This could be changed for testing purposes, e.g. to use only 1% of the dataset, set this argument to `400000`. This script returns the preprocessed dataset ready to be used as input for the TreeNiN: 
    
    -  Loads and reclusters the jet constituents. 
    - Creates binary trees with the clustering history of the jets and outputs a dictionary for each jet that contains the root_id, tree, content (constituents 4-momentum vectors), mass, pT, energy, eta and phi values. 
    - Preprocessing is applied. The initial 7 features are: p, eta, phi, E, E/JetE, pT, theta.
    
    The final preprocessed test dataset is saved here: [`test_top_tag_reference_dataset_kt_1200000_Njets_.pkl`](/code/data/preprocessed_trees/test_top_tag_reference_dataset_kt_1200000_Njets_.pkl)

2. Run the TreeNiN workflow (loading previously trained weights):
    `python MLWorkflow.py 0 9`
    
    where we are testing over all the models between `0` and `9`.
    
3. Read and save the output probabilities for all the models:
    `python3 saveProb.py`
    
    This saves the probabilities here: [`TreeNiN_hd50.pkl`](code/top_reference_dataset/outProb/TreeNiN_hd50.pkl)
   

#### Training mode:

##### **Data pipeline**
To run on training mode we need to preprocess the train and val datasets as well. Follow the instructions below and rerun for each dataset (train, val, test).

1. Modify the last two lines of [`ReadData.py`](code/top_reference_dataset/ReadData.py)  to load the validation (train) dataset. (You would first need to download the validation and training datasets from [here](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6) )
2. `cd `[`TreeNiN_code/top_reference_dataset/`](code/top_reference_dataset/)
3. `python3 ReadData.py 0`
4. `python2.7 toptag_reference_dataset_Tree.py jet_image_trim_pt800-900_card.dat test out_data/ ../data/inputTrees/top_tag_reference_dataset/`  
(The output file with the dictionary for each jet will be saved in [`data/inputTrees/top_tag_reference_dataset/`](code/data/inputTrees/top_tag_reference_dataset/). Also, change `test` for `val` or `train` to run the data pipeline over the validation and train sets.)
5. Set the flag *test_train_val_datasets=True* in [`preprocess_main.py`](code/recnn/preprocess_main.py).
6. `cd `[`TreeNiN_code/recnn/`](code/recnn/)
7. `python3 `[`run_preprocess.py`](code/recnn/run_preprocess.py)


##### **TreeNiN**

1. Set the flag *TRAIN_and_EVALUATE=True* and *EVALUATE=False* in  [`search_hyperparams.py`](code/recnn/search_hyperparams.py) (This will run [`train.py`](code/recnn/train.py) and [`evaluate.py`](code/recnn/evaluate.py)). 
2. Specify the *multi_scan* function arguments as explained below.
3. Run: `python3 search_hyperparams.py --gpu=0`. Results will be saved in `experiments/[dir_name]`.

-------------------------------------------------------------------------
## Running directly


### Dependencies

Make sure the following tools are installed and running:

- Python 2.7, Python 3.6, packages included in [Anaconda](https://www.anaconda.com/) (Numpy, Scipy, scikit-learn, Pandas),  [FastJet](http://fastjet.fr/), [PyTorch](https://pytorch.org/).



-------------------------------------------------------------------------
## Data processing structure

The data dir is [`top_reference_dataset`](code/top_reference_dataset):

-[`ReadData.py`](code/top_reference_dataset/ReadData.py): Load the datasets in h5 format, get the labels and non-zero values for the jet constituents and save pickle files in [`out_data`](code/top_reference_dataset/out_data).

-[`toptag_reference_dataset_Tree.py`](code/top_reference_dataset/toptag_reference_dataset_Tree.py): Load and recluster the jet constituents. Create binary trees with the clustering history of the jets and output a dictionary for each jet that contains the root_id, tree, content (constituents 4-momentum vectors), mass, pT, energy, eta and phi values (also charge, muon ID, etc depending on the information contained in the dataset). Auxiliary scripts that are called:

- [`analysis_functions.py`](code/top_reference_dataset/analysis_functions.py): auxiliary functions. 
- [`preprocess_functions.py`](code/top_reference_dataset/preprocess_functions.py): auxiliary functions. 
- [`tree_cluster_hist.py`](code/top_reference_dataset/tree_cluster_hist.py): Create binary trees. If the dataset has more information besides the jet constituents 4-vectors (e.g. charge, muon ID, etc), this script runs a recursive function that traverses the tree down to the leaves and then goes back to the root generating the inner nodes values by adding the children values for the extra information.
- [`jet_image_trim_pt800-900_card.dat`](code/top_reference_dataset/jet_image_trim_pt800-900_card.dat): parameters card with kinematic variables values, clustering algorithm choice, etc. Currently not used.




-------------------------------------------------------------------------
## TreeNiN structure

The TreeNiN code dir is [`recnn`](code/recnn):

- [`search_hyperparams.py`](code/recnn/search_hyperparams.py): main script that calls preprocess_main.py, train.py and evaluate.py; and runs hyperparameters searches.
- Parameters to specify before running:
- *multi_scan* function arguments. Determine the hyperparameter values (for a scan input a list of values), *dir_name* and *number of runs=Nrun_finish-Nrun_start*.
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
`python3 search_hyperparams.py --gpu=0`

- [`run_preprocess.py`](code/recnn/run_preprocess.py): same as search_hyperparams.py but hard-coded to do the preprocessing only.

- [`preprocess_main.py`](code/recnn/preprocess_main.py): Preprocess the jet dictionaries and go from the jet constituents 4-momentum vectors to the 7 features for the nodes (|p|, eta, phi, E, E/Ejet, pT and theta). Before running, define *toptag_reference_train*, *toptag_reference_val* and *toptag_reference_test* with the  name of the input trees saved in [`data/inputTrees/top_tag_reference_dataset/`](code/data/inputTrees/top_tag_reference_dataset/) (e.g. `toptag_reference_train=dir_jets_subjets+'/tree_train_jets.pkl'`)


- [`train.py`](code/recnn/train.py): Train the model. This file calls `model/recNet.py`, `model/data_loader.py`, `model/preprocess.py`, `model/dataset.py` and `utils.py`

- [`evaluate.py`](code/recnn/evaluate.py): loads the weights that give the best val accuracy (this could be changed to lowest val loss function value) and gets the accuracy, tpr, fpr, and ROC auc over the test set.

- [`model`](code/recnn/model/):

- [`recNet.py`](code/recnn/model/recNet.py): model architecture for batch training and accuracy function.

- [`data_loader.py`](code/recnn/model/data_loader.py): load the raw data and create the batches:

- Load the jet events and make the trees.
- Shuffle the signal and background sets independently. Split the sample into train, validation and test with equal number of sg and bg events. Then shuffle each set.
- Load the jet trees, reorganize the tree by levels, create a batch of N jets by appending the nodes of each jet to each level and add zero padding so that all the levels have the same size
- Generator function that loads the batches, shifts numpy arrays to torch tensors and feeds the training/validation pipeline


- [`preprocess.py`](code/recnn/model/preprocess.py): rewrite and reorganize the jet contents (e.g. add features for each node such as energy, pT, eta, phi, charge, muon ID, etc) 

- [`dataset.py`](code/recnn/model/dataset.py): This script defines:
- A new subclass of torch.utils.data.Dataset that overrides the *__len__*  and *__getitem__* methods. This allows to call torch.utils.data.DataLoader to generate the batches in parallel with num_workers>1 and speed up the code. 
- A customized *collate* function to load the batches with torch.utils.data.DataLoader.

-[`experiments`](code/recnn/experiments):

- [`template_params.json`](code/recnn/experiments/template_params.json):  Template file that contains all the architecture parameters and training hyperparameters for a specific run. “search_hyperparams.py” modifies these parameters for each scan

- [`experiments/[dir_name]`](code/recnn/experiments): dir with all the hyperparameter scan results (weights, log files, results) for each sample/architecture. For each run, a new directory will be created with files saving the output probabilities on the test set, metrics history after each epoch, best and last weights, etc.

-[`jet_study.ipynb`](code/recnn/jet_study.ipynb): Load results from `experiments/[dir_name]`, and get results (accuracy, AUC, background rejection, etc) for single and/or multiple runs.

-[`utils.py`](code/recnn/utils.py): auxiliary functions for training, logging, loading hyperparameters from json file, etc.






-------------------------------------------------------------------------

## Acknowledgements

SM gratefully acknowledges the support of NVIDIA Corporation with the donation of a Titan V GPU used for this research.

-------------------------------------------------------------------------
## References

If you use this package, please cite this code as

```
@misc{TreeNiN,
author       = "Macaluso, Sebastian and Cranmer, Kyle",
title        = "{Tree Network in Network (TreeNiN) for jet physics}",
note         = "{DOI: 10.5281/zenodo.2582216}",
year         = {2019},
url          = {https://github.com/SebastianMacaluso/TreeNiN}
}
```



















