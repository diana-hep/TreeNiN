"""Train the model"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from sklearn.utils import check_random_state
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import time
import pickle

import utils
import model.data_loader as dl
from model import recNet as net
from model import preprocess 


#-------------------------------------------------------------------------------------------------------------
#/////////////////////    TRAINING AND EVALUATION FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network superclass
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    ##-----------------------------
    # Use tqdm for progress bar
    t = trange(num_steps) 
    for i in t:
    
        # fetch the next training batch   
        levels, children, n_inners, contents, n_level, labels_batch = next(data_iterator)
        output_batch = model(params, levels, children, n_inners, contents, n_level)

        # compute model output and loss
        labels_batch = labels_batch.float()  #Uncomment if using torch.nn.BCELoss() loss function
        output_batch=output_batch.view((params.batch_size)) # For 1 final neuron 
        loss = loss_fn(output_batch, labels_batch)
        
#         print('output_batch=',output_batch)
#         print('labels_batch=',labels_batch)
#         print('y_pred=',output_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        
        ##-----------------------------
        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
  
        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg())) #Uncomment once tqdm is installed
     
#     print('summ=',summ)    
    ##-----------------------------
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
#     print('metrics_mean=',metrics_mean)
#     print('metrics_string=',metrics_string)
    
#-------------------------------------------------------------------------------------------------------------
def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network superclass
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    ##-----------------------------
    # compute metrics over the dataset
    for _ in range(num_steps):
    
        # fetch the next evaluation batch
        levels, children, n_inners, contents, n_level, labels_batch = next(data_iterator)
        output_batch = model(params, levels, children, n_inners, contents, n_level)

        # compute model output
        labels_batch = labels_batch.float() #Uncomment if using torch.nn.BCELoss() loss function
        output_batch=output_batch.view((params.batch_size)) # For 1 final neuron 
        loss = loss_fn(output_batch, labels_batch)
#         print('labels for loss=',labels_batch)
#         print('y_pred=',output_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
#         summary_batch['loss'] = loss.data[0]
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
        
    ##-----------------------------
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

#-------------------------------------------------------------------------------------------------------------
def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, step_size, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network superclass
        train_data: array with levels, children, n_inners, contents, n_level and labels_batch lists
        val_data: array levels, children, n_inners, contents, n_level and labels_batch lists
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log files
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_acc = 0.0

    ##-----------------------------
    
    
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # define the data iterator to feed the data and call the training function
        train_data_iterator = data_loader.make_pad_batch_iterator_level(train_data, params.batch_size)
        train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps_train)
            
        # Evaluate for one epoch on validation set
        val_data_iterator = data_loader.make_pad_batch_iterator_level(val_data, params.batch_size)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps_val)
        
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc
        
        scheduler.step()
        step_size = step_size * decay
        

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=model_dir)
            
        # If best_eval, best_save_path        
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
    
#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
if __name__=='__main__':  

  ##----------------------------------------------------------------------------------------------------------
  # Global variables
  ##-----------------
  data_dir='../data/'
  os.system('mkdir -p '+data_dir)
  
  # Select the right dir for jets data
  batches_dir='input_batches_pad/'
  os.system('mkdir -p '+data_dir+'/'+batches_dir)
  
  ##-----------------
  #If true the batchization of the data is generated. Do it only once and the turn in off
  make_batch=True
#   make_batch=False

#   pT_order=True
  pT_order=False

  # Select the input sample
#   nyu=True
  nyu=False
  
  ##-----------------  
  if nyu==True:
    #Directory with the input trees
    sample_name='nyu_jets'
    
  #   algo='antikt-antikt-delphes'
#     algo='antikt-kt-delphes'
    algo='antikt-antikt'
  
    file_name=algo+'-train.pickle'
  else:
    algo=''
    
    #Directory with the input trees
    ### CHECK THAT SEARCH_HYPERPARAMS.PY HAS THE SAME SAMPLE NAME
    
#     sample_name='top_qcd_jets_antikt_antikt'
#     sample_name='top_qcd_jets_antikt_kt'
    sample_name='top_qcd_jets_antikt_CA'
    
    #labels to look for the input files
  #   sg='tt'
    sg='ttbar' 
    bg='qcd'
  

  
  ##------------------------------------------------------------  
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
  parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
  parser.add_argument('--restore_file', default=None,
                      help="Optional, name of the file in --model_dir containing weights to reload before \
                      training")  # 'best' or 'last'
  
  
  # Load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)

  ##-----------------
  # Set the logger
  utils.set_logger(os.path.join(args.model_dir, 'train.log'))
  
  dir_jets_subjets= args.data_dir
  

  # Output file names with the batches of data. We rewrite the sample name if running from search_hyperparam.py
  sample_name=str(args.data_dir).split('/')[-1]

  logging.info('sample_name={}'.format(sample_name))
  logging.info('----'*20)
#   batch_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'
  batch_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'+'_'+str(params.info)
  logging.info('batch_filename={}'.format(batch_filename))
  train_data=data_dir+batches_dir+'train_'+batch_filename+'.pkl'
  val_data=data_dir+batches_dir+'dev_'+batch_filename+'.pkl'
  test_data=data_dir+batches_dir+'test_'+batch_filename+'.pkl'
  
  
  transformer_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'
  transformer_data=data_dir+batches_dir+'transformer_'+transformer_filename+'.pkl'
  
  start_time = time.time()
  
  ##-----------------------------------------------------------------------------------------------
  data_loader=dl.DataLoader # Main class with the methods to load the raw data and create the batches
  
  # Create the batches (do it only once, then turn make_batch=False)
  if make_batch==True:
    
    if nyu==True: # We load the nyu samples
        
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
      X = [preprocess.extract(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in X]
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

    ##---------------------
    # Here we use our own samples
    else:
      # loading dataset_params and make trees
      sig_list=data_loader.makeTrees(dir_jets_subjets,sg,params.myN_jets,1)
      bkg_list=data_loader.makeTrees(dir_jets_subjets,bg,params.myN_jets,0) 
      elapsed_time=time.time()-start_time
      logging.info('Tree generation time (minutes) ={}'.format(elapsed_time/60))
    
      # Merge sg and bg into an equal size sample and shuffle the sets
#       X,Y= data_loader.merge_shuffle_sample(sig_list,bkg_list)
    
      # Preprocessing steps: Ensure that the left sub-jet has always a larger pt than the right. Change the input variables (features)
      if pT_order==True:
        sig_list = [preprocess.extract(preprocess.sequentialize_by_pt(preprocess.permute_by_pt(jet)), params.features) for jet in sig_list]
        bkg_list = [preprocess.extract(preprocess.sequentialize_by_pt(preprocess.permute_by_pt(jet)), params.features) for jet in bkg_list]       
      
      else:
        sig_list = [preprocess.extract(preprocess.permute_by_pt(jet), params.features,kappa=0.4) for jet in sig_list]
        bkg_list = [preprocess.extract(preprocess.permute_by_pt(jet), params.features,kappa=0.4) for jet in bkg_list]      

#         sig_list = [preprocess.extract(preprocess.permute_by_pt(jet), params.features) for jet in sig_list]
#         bkg_list = [preprocess.extract(preprocess.permute_by_pt(jet), params.features) for jet in bkg_list]

      
#       sig_list = [preprocess.extract(preprocess.permute_by_pt(preprocess.rewrite_content(jet)), params.features) for jet in sig_list]
#       bkg_list = [preprocess.extract(preprocess.permute_by_pt(preprocess.rewrite_content(jet)), params.features) for jet in bkg_list]     
     
      # Split into train+validation+test and shuffle
      logging.info("Splitting into train, validation and test datasets, and shuffling...")
      train_x, train_y, dev_x, dev_y, test_x, test_y = data_loader.split_shuffle_sample(sig_list, bkg_list, 0.6, 0.2, 0.2)


#       train_x, X_valid, train_y, Y_valid = train_test_split(X, Y, test_size=0.4, random_state=0)
# 
#       dev_x, test_x, dev_y, test_y = train_test_split(X_valid, Y_valid, test_size=0.5, random_state=1)

      
      # Apply RobustScaler (remove outliers, center and scale data)
      transformer=data_loader.get_transformer(train_x)
      # Save transformer
      with open(transformer_data, "wb") as f: pickle.dump(transformer, f)
      
      
      #Scale features using the training set transformer
      train_x = data_loader.transform_features(transformer,train_x)
      dev_x = data_loader.transform_features(transformer,dev_x)
      test_x = data_loader.transform_features(transformer,test_x)

        
        

    elapsed_time=time.time()-start_time
    logging.info('Split sample time (minutes) ={}'.format(elapsed_time/60))
    

    ##-----------------------------------------------------------------------------------------------
    # Generate dataset batches. 
    logging.info("Creating train batches ...")    
    train_batches=dl.batch_array(train_x, train_y, params.batch_size, params.features)
    logging.info("Creating val batches ...")
    dev_batches=dl.batch_array(dev_x, dev_y, params.batch_size, params.features)
    if nyu==False:
      logging.info("Creating test batches ...")
      test_batches=dl.batch_array(test_x, test_y, params.batch_size, params.features)
      logging.info('Number of test_batches={}'.format(len(test_batches)))

    elapsed_time=time.time()-start_time
    logging.info('Batch generation time (minutes) ={}'.format(elapsed_time/60))
    logging.info('Train and val batch size={}'.format(params.batch_size))
    logging.info('Number of train_batches={}'.format(len(train_batches)))
    logging.info('Number of val_batches={}'.format(len(dev_batches)))

    # Save batches
    with open(train_data, "wb") as f: pickle.dump(train_batches, f)
    with open(val_data, "wb") as f: pickle.dump(dev_batches, f)
    if nyu==False:
      with open(test_data, "wb") as f: pickle.dump(test_batches, f)

#   sys.exit()
  ##------------------------------------------------------------------------------------------------
  ###   TRAINING
  ##------------------------------------------------------------------------------------------------
  # use GPU if available
  params.cuda = torch.cuda.is_available()
  
  # Set the random seed for reproducible experiments
  torch.manual_seed(230)
  if params.cuda: torch.cuda.manual_seed(230)

  ##-----------------------------
  # Create the input data pipeline 
  logging.info('---'*20)
  logging.info("Loading the datasets...")

  # Load data 
  with open(train_data, "rb") as f: train_data=pickle.load(f)
  with open(val_data, "rb") as f: val_data=pickle.load(f) 
     
  # specify the train and val dataset sizes
  num_steps_train=len(train_data)
  num_steps_val=len(val_data)

  logging.info("- done loading the datasets") 
  logging.info('---'*20)   
  
  ##---------------------------------------------------------------------
  ''' OPTIMIZER AND LOSS '''
  '''----------------------------------------------------------------------- '''
  # Define the model and optimizer
  model = net.GRNNTransformSimple_level(params).cuda() if params.cuda else net.GRNNTransformSimple_level(params)   

  logging.info("Model= {}".format(model))
  logging.info("---"*20)
  step_size=params.learning_rate
  decay=params.decay
  
  logging.info("Building optimizer...")
#   optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=step_size)#,eps=1e-05)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

  # fetch loss function and metrics
  loss_fn = torch.nn.BCELoss()
#   loss_fn = torch.nn.CrossEntropyLoss()
  metrics = net.metrics

  ##----------------------
  # Train the model
  logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

  train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir, step_size,
                     args.restore_file)   
  
  elapsed_time=time.time()-start_time
  logging.info('Total time (minutes) ={}'.format(elapsed_time/60))
    
    


    
    
    
  
