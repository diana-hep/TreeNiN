"""Train the model"""
"""This is from the pytorch_shuffle dir"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import time
import pickle
import utils
import model.data_loader as dl
import model.dataset as dataset
from model import recNet as net
from model import preprocess 

##----------------------------------------------------------------------------------------------------------
if __name__=='__main__':  

  print('Preprocessing jet trees ...')
  print('==='*20)
  ##----------------------------------------------------------------------------------------------------------
  # Global variables
  ##-------------------
  data_dir='../data/'
  os.system('mkdir -p '+data_dir)
  
  # Select the right dir for jets data
  trees_dir='preprocessed_trees/'
  os.system('mkdir -p '+data_dir+'/'+trees_dir)
  
  ##-------------------
  #If true the preprocessed trees are generated and saved. Do it only once and then turn in off
#   make_preprocess=True
#   make_preprocess=False

#   pT_order=True
  pT_order=False

  # Select the input sample
#   nyu=True
  nyu=False
  
  toptag_reference_dataset=True # If True, specify below the location of the train, val, test input trees  
  test_train_val_datasets=False
  
  ##--------------------  
  algo=''
  sample_name=''
#   if nyu==True:
    #Directory with the input trees
#     sample_name='nyu_jets'
    
  #   algo='antikt-antikt-delphes'
#     algo='antikt-kt-delphes'
#     algo='antikt-antikt'
    
    # file_name=algo+'-train.pickle'
    
#   if toptag_reference_dataset  
#     sample_name='top_tag_reference_dataset'
    
    
#   else:
#     algo=''
    
    #Directory with the input trees
    ### CHECK THAT SEARCH_HYPERPARAMS.PY HAS THE SAME SAMPLE NAME
    
#     sample_name='top_qcd_jets_antikt_antikt'
#     sample_name='top_qcd_jets_antikt_kt'
#     sample_name='top_qcd_jets_antikt_CA'
    
    #labels to look for the input files
  #   sg='tt'
  
  sg='ttbar' 
  bg='qcd'
  

  
  ##------------------------------------------------------------  
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
  parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
#   parser.add_argument('--restore_file', default=None,
#                       help="Optional, name of the file in --model_dir containing weights to reload before \
#                       training")  # 'best' or 'last'
  
  parser.add_argument('--jet_algorithm', default=algo, help="jet algorithm")
#   parser.add_argument('--transformer', default='../data/preprocessed_trees/', help="Transformen from running RobustScaler on the train dataset")
  
  # Load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)

  ##-------------------
  # Set the logger
  utils.set_logger(os.path.join(args.model_dir, 'preprocess.log'))
  
  dir_jets_subjets= args.data_dir
  algo=args.jet_algorithm
  file_name=algo+'-train.pickle'
  ##-------------------
  # Define output file names with the batches of data. We rewrite the sample name if running from search_hyperparam.py
  sample_name=str(args.data_dir).split('/')[-1]
  logging.info('sample_name={}'.format(sample_name))
  logging.info('----'*20)
  # sample_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'+'_'+str(params.info)
  sample_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.info)
  logging.info('sample_filename={}'.format(sample_filename))
  
  train_data=data_dir+trees_dir+'train_'+sample_filename+'.pkl'
  val_data=data_dir+trees_dir+'dev_'+sample_filename+'.pkl'
  test_data=data_dir+trees_dir+'test_'+sample_filename+'.pkl'
    
  transformer_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'
  transformer_data=data_dir+trees_dir+'transformer_'+transformer_filename+'.pkl'
  
  start_time = time.time()  
  ##-----------------------------------------------------------------------------------------------------------
  data_loader=dl.DataLoader # Main class with the methods to load the raw data, create and preprocess the trees
  
  # Create the batches 
#   if make_preprocess==True:
  
  # FOR NYU SAMPLES (from arXiv:1702.00748)
  if nyu==True: 
      
    nyu_train=dir_jets_subjets+'/'+file_name
    # loading dataset_params and make trees
    logging.info('nyu_train={}'.format(nyu_train))
    fd = open(nyu_train, "rb")
    X, Y = pickle.load(fd,encoding='latin-1')
    fd.close()
    
    X=np.asarray(X)
    Y=np.asarray(Y)
    
    logging.info('Training data size={}'.format(len(X)))
    logging.info('---'*20)
    
    # Shuffle the sets
    indices = check_random_state(1).permutation(len(X))
    X = X[indices]
    Y = Y[indices]  
    X=np.asarray(X)
    Y=np.asarray(Y)
    
    # Preprocessing steps: Ensure that the left sub-jet has always a larger pt than the right. Change the input variables (features)
    X = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in X]
# 
#       # Apply RobustScaler (remove outliers, center and scale data)
#       X=data_loader.scale_features(X) 

    # Split into train+validation
    logging.info("Splitting into train and validation...")
    
    train_x, dev_x, train_y, dev_y = train_test_split(X, Y, test_size=5000, random_state=0)

    # Apply RobustScaler (remove outliers, center and scale data)
    transformer=data_loader.get_transformer(train_x)
    # Save transformer
    with open(transformer_data, "wb") as f: pickle.dump(transformer, f)
    
    
    #Scale features using the training set transformer
    train_x = data_loader.transform_features(transformer,train_x)
    dev_x = data_loader.transform_features(transformer,dev_x)



    #----------------------------------------------------------
    # Preprocess test set
    
    # Create the input data pipeline
    logging.info("Preprocessing the test dataset...")
    
#     batches_dir='input_batches_pad/'
#     test_data=data_dir+batches_dir+'test_'+sample_filename+'.pkl'
    

    #Load test smaple
    nyu_test=dir_jets_subjets+'/'+file_name
    print('nyu_test=',nyu_test)
    fd = open(nyu_test, "rb")
    X, y = pickle.load(fd,encoding='latin-1')
    fd.close()

    X=np.asarray(X)
    y=np.asarray(y)
    
    logging.info('Training data size= '+str(len(X)))
    logging.info('---'*20)
    
    #------------------
    # Shuffle the sets
    indices = check_random_state(1).permutation(len(X))
    X = X[indices]
    y = y[indices]  
    X=np.asarray(X)
    y=np.asarray(y)
    
  # Preprocessing steps: Ensure that the left sub-jet has always a larger pt than the right. Change the input variables (features)
    X = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in X]


    # Apply RobustScaler (remove outliers, center and scale data) with train set transformer
#     transformer_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'
#     transformer_data=data_dir+batches_dir+'transformer_'+transformer_filename+'.pkl'      
    with open(transformer_data, "rb") as f: transformer =pickle.load(f)
    
    #Scale features using the training set transformer
    X = data_loader.transform_features(transformer,X)


#   
#       # Apply RobustScaler (remove outliers, center and scale data)     
#       X=data_loader.scale_features(X) 

    #------------------
    # Cropping
    X_ = [j for j in X if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]
    y_ = [y[i] for i, j in enumerate(X) if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]

    X = X_
    y = y_
    X=np.asarray(X)
    y = np.asarray(y)
  
    logging.info('Lenght X= '+str(len(X)))
    logging.info('Length y= '+str(len(y)))
    
    #------------------
    # Weights for flatness in pt
    w = np.zeros(len(y))    
  
    logging.info('Length w before='+str(len(w)))
    
    X0 = [X[i] for i in range(len(y)) if y[i] == 0]
    pdf, edges = np.histogram([j["pt"] for j in X0], density=True, range=[250, 300], bins=50)
    pts = [j["pt"] for j in X0]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y==0] = inv_w
      
    X1 = [X[i] for i in range(len(y)) if y[i] == 1]
    pdf, edges = np.histogram([j["pt"] for j in X1], density=True, range=[250, 300], bins=50)
    pts = [j["pt"] for j in X1]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y==1] = inv_w

    logging.info('Length w after='+str(len(w)))


    test_x=X
    test_y=y


    # Save trees
    with open(train_data, "wb") as f: pickle.dump(zip(train_x,train_y), f)
    with open(val_data, "wb") as f: pickle.dump(zip(dev_x,dev_y), f)
    with open(test_data, "wb") as f: pickle.dump((zip(test_x,test_y),w), f) #We save the weights and test set

    
#     
#     #-------------------
#     # Generate dataset batches.
#     test_batches=dl.batch_array(X, y, params.batch_size, params.features)
#     logging.info('Number of test_batches='+str(len(test_batches)))  
# 
#     # Save batches
#     with open(test_data, "wb") as f: pickle.dump((test_batches,w), f)
    
    logging.info("- done.")

  ##-------------------------------------------------------------------------------------------------
  ##-------------------------------------------------------------------------------------------------
  if toptag_reference_dataset:

    if test_train_val_datasets:
    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets_1200000_R_0.3_rot_boost_rot_flip.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets_400000_R_0.3_rot_boost_rot_flip.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets_400000_R_0.3_rot_boost_rot_flip.pkl'

    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets_120001.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets_40001.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets_40001.pkl'
  
        toptag_reference_train=dir_jets_subjets+'/tree_train_jets.pkl'
        toptag_reference_val=dir_jets_subjets+'/tree_val_jets.pkl'
        toptag_reference_test=dir_jets_subjets+'/tree_test_jets.pkl'

    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets_1001.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets_1001.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets_1001.pkl'
    
        # loading dataset_params and make trees
        logging.info('Loading toptag_reference_dataset={}'.format(toptag_reference_val))
        with open(toptag_reference_val, "rb") as f: toptag_reference_val =pickle.load(f,encoding='latin-1')  
  
        logging.info('Loading toptag_reference_dataset={}'.format(toptag_reference_train))
        with open(toptag_reference_train, "rb") as f: toptag_reference_train =pickle.load(f,encoding='latin-1')  
    
        logging.info('Loading toptag_reference_dataset={}'.format(toptag_reference_test))   
        with open(toptag_reference_test, "rb") as f: toptag_reference_test =pickle.load(f,encoding='latin-1') 
    #   

        toptag_reference_train_x=np.asarray([x for (x,y) in toptag_reference_train])
        toptag_reference_train_y=np.asarray([y for (x,y) in toptag_reference_train])
    
        toptag_reference_val_x=np.asarray([x for (x,y) in toptag_reference_val])
        toptag_reference_val_y=np.asarray([y for (x,y) in toptag_reference_val])
        
        toptag_reference_test_x=np.asarray([x for (x,y) in toptag_reference_test])
        toptag_reference_test_y=np.asarray([y for (x,y) in toptag_reference_test])
  
        logging.info('Training data size={}'.format(len(toptag_reference_train_x)))
        logging.info('---'*20)
    
        # Shuffle the training set
        indices = check_random_state(1).permutation(len(toptag_reference_train_x))
        toptag_reference_train_x = toptag_reference_train_x[indices]
        toptag_reference_train_y = toptag_reference_train_y[indices]  
    
        toptag_reference_train_x=np.asarray(toptag_reference_train_x)
        toptag_reference_train_y=np.asarray(toptag_reference_train_y)
    
    
    #     print('toptag_reference_train_y=',toptag_reference_train_y[0:20])

        # Preprocess
        toptag_reference_train_x = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in toptag_reference_train_x]
    
        toptag_reference_val_x = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in toptag_reference_val_x]
    
        toptag_reference_test_x = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in toptag_reference_test_x]


        # Apply RobustScaler (remove outliers, center and scale data)
        transformer=data_loader.get_transformer(toptag_reference_train_x)
        # Save transformer
        with open(transformer_data, "wb") as f: pickle.dump(transformer, f)
    
    
        #Scale features using the training set transformer
        toptag_reference_train_x = data_loader.transform_features(transformer,toptag_reference_train_x)
        toptag_reference_val_x = data_loader.transform_features(transformer,toptag_reference_val_x)
        toptag_reference_test_x = data_loader.transform_features(transformer,toptag_reference_test_x)


        ##---------------------------------  
  
        # Save trees
        with open(train_data, "wb") as f: pickle.dump(zip(toptag_reference_train_x,toptag_reference_train_y), f)
        with open(val_data, "wb") as f: pickle.dump(zip(toptag_reference_val_x,toptag_reference_val_y), f)
        with open(test_data, "wb") as f: pickle.dump(zip(toptag_reference_test_x,toptag_reference_test_y), f)    
    
    
    else:
        toptag_reference_test=dir_jets_subjets+'/tree_test_jets.pkl'
    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets_1200000_R_0.3_rot_boost_rot_flip.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets_400000_R_0.3_rot_boost_rot_flip.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets_400000_R_0.3_rot_boost_rot_flip.pkl'

    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets_120001.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets_40001.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets_40001.pkl'
  
    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets.pkl'

    #     toptag_reference_train=dir_jets_subjets+'/tree_train_jets_1001.pkl'
    #     toptag_reference_val=dir_jets_subjets+'/tree_val_jets_1001.pkl'
    #     toptag_reference_test=dir_jets_subjets+'/tree_test_jets_1001.pkl'
    
        # loading dataset_params and make trees
    #     logging.info('Loading toptag_reference_dataset={}'.format(toptag_reference_val))
    #     with open(toptag_reference_val, "rb") as f: toptag_reference_val =pickle.load(f,encoding='latin-1')  
    #   
    #     logging.info('Loading toptag_reference_dataset={}'.format(toptag_reference_train))
    #     with open(toptag_reference_train, "rb") as f: toptag_reference_train =pickle.load(f,encoding='latin-1')  
    
        logging.info('Loading toptag_reference_dataset={}'.format(toptag_reference_test))   
        with open(toptag_reference_test, "rb") as f: toptag_reference_test =pickle.load(f,encoding='latin-1') 
    #   

    #     toptag_reference_train_x=np.asarray([x for (x,y) in toptag_reference_train])
    #     toptag_reference_train_y=np.asarray([y for (x,y) in toptag_reference_train])
    #     
    #     toptag_reference_val_x=np.asarray([x for (x,y) in toptag_reference_val])
    #     toptag_reference_val_y=np.asarray([y for (x,y) in toptag_reference_val])
        
        toptag_reference_test_x=np.asarray([x for (x,y) in toptag_reference_test])
        toptag_reference_test_y=np.asarray([y for (x,y) in toptag_reference_test])
  
    #     logging.info('Training data size={}'.format(len(toptag_reference_train_x)))
        logging.info('---'*20)
    
        # Shuffle the training set
    #     indices = check_random_state(1).permutation(len(toptag_reference_train_x))
    #     toptag_reference_train_x = toptag_reference_train_x[indices]
    #     toptag_reference_train_y = toptag_reference_train_y[indices]  
    #     
    #     toptag_reference_train_x=np.asarray(toptag_reference_train_x)
    #     toptag_reference_train_y=np.asarray(toptag_reference_train_y)
    
    
    #     print('toptag_reference_train_y=',toptag_reference_train_y[0:20])

        # Preprocess
    #     toptag_reference_train_x = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in toptag_reference_train_x]
    #     
    #     toptag_reference_val_x = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in toptag_reference_val_x]
    
        toptag_reference_test_x = [preprocess.extract_nyu_samples(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in toptag_reference_test_x]


        # Apply RobustScaler (remove outliers, center and scale data)
    #     transformer=data_loader.get_transformer(toptag_reference_train_x)
        # Save transformer
    #     with open(transformer_data, "wb") as f: pickle.dump(transformer, f)
    
        #Load transformer
        with open(transformer_data, "rb") as f: transformer =pickle.load(f) 
    
        #Scale features using the training set transformer
    #     toptag_reference_train_x = data_loader.transform_features(transformer,toptag_reference_train_x)
    #     toptag_reference_val_x = data_loader.transform_features(transformer,toptag_reference_val_x)
        toptag_reference_test_x = data_loader.transform_features(transformer,toptag_reference_test_x)


        ##---------------------------------  
  
        # Save trees
    #     with open(train_data, "wb") as f: pickle.dump(zip(toptag_reference_train_x,toptag_reference_train_y), f)
    #     with open(val_data, "wb") as f: pickle.dump(zip(toptag_reference_val_x,toptag_reference_val_y), f)
        with open(test_data, "wb") as f: pickle.dump(zip(toptag_reference_test_x,toptag_reference_test_y), f)    
    
  
  ##-------------------------------------------------------------------------------------------------
  # FOR OUR OWN SAMPLES
  else:
  
    # loading dataset_params and make trees
    sig_list=data_loader.makeTrees(dir_jets_subjets,sg,params.myN_jets,1)
    bkg_list=data_loader.makeTrees(dir_jets_subjets,bg,params.myN_jets,0) 
    elapsed_time=time.time()-start_time
    logging.info('Tree generation time (minutes) ={}'.format(elapsed_time/60))
  
    ##----------------------------------------------
    # Preprocessing steps: 
    #  - Ensure that the left sub-jet has always a larger pt than the right (or more leaves). 
    #  - Change the input variables (features)
    
    if pT_order==True:
      sig_list = [preprocess.extract(preprocess.sequentialize_by_pt(preprocess.permute_by_pt(jet)), params.features) for jet in sig_list]
      bkg_list = [preprocess.extract(preprocess.sequentialize_by_pt(preprocess.permute_by_pt(jet)), params.features) for jet in bkg_list]       
    
    else:
    
      ##----
      ## a) Add number of jet constituents in each branch and assign the branch with the largest number as the left node
#         sig_list = [preprocess.extract(preprocess.permute_by_n_leaves(preprocess.rewrite_content_leaves(jet)), params.features,kappa=0.4) for jet in sig_list]
#         bkg_list = [preprocess.extract(preprocess.permute_by_n_leaves(preprocess.rewrite_content_leaves(jet)), params.features,kappa=0.4) for jet in bkg_list]      
    
      ##----
      ## b) Assign the biggest pT subjet as the left node
    
      sig_list = [preprocess.extract(preprocess.permute_by_pt(jet), params.features,kappa=0.4) for jet in sig_list]
      bkg_list = [preprocess.extract(preprocess.permute_by_pt(jet), params.features,kappa=0.4) for jet in bkg_list]      
  
  
    ##-------------------
    # Split into train+validation+test and shuffle
    logging.info("Splitting into train, validation and test datasets, and shuffling...")
    train_x, train_y, dev_x, dev_y, test_x, test_y = data_loader.split_shuffle_sample(sig_list, bkg_list, 0.6, 0.2, 0.2)


#       train_x, X_valid, train_y, Y_valid = train_test_split(X, Y, test_size=0.4, random_state=0)
#       dev_x, test_x, dev_y, test_y = train_test_split(X_valid, Y_valid, test_size=0.5, random_state=1)

    ##-------------------
    # Apply RobustScaler (remove outliers, center and scale data)
    transformer=data_loader.get_transformer(train_x)
    
    # Save transformer
    with open(transformer_data, "wb") as f: pickle.dump(transformer, f)
    
    #Scale features using the training set transformer
    train_x = data_loader.transform_features(transformer,train_x)
    dev_x = data_loader.transform_features(transformer,dev_x)
    test_x = data_loader.transform_features(transformer,test_x)

        
    ##---------------------------------   
    elapsed_time=time.time()-start_time
    logging.info('Split sample time (minutes) ={}'.format(elapsed_time/60))
  
    # Save trees
    with open(train_data, "wb") as f: pickle.dump(zip(train_x,train_y), f)
    with open(val_data, "wb") as f: pickle.dump(zip(dev_x,dev_y), f)
  #   if nyu==True:
  #     #We save the weights
  #     with open(test_data, "wb") as f: pickle.dump((zip(test_x,test_y),w), f)
  #   else:
    with open(test_data, "wb") as f: pickle.dump(zip(test_x,test_y), f)





    
    
    
  
