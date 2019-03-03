"""Train the model"""
"""This is from the pytorch_shuffle dir"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import tqdm
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import time
import pickle
import utils
import model.data_loader as dl
import model.dataset as dataset
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
    
    data_iterator_iter = iter(data_iterator)
    
    
    for i in t:
    
        time_before_batch=time.time() 
        
        # fetch the next training batch
        levels, children, n_inners, contents, n_level, labels_batch=next(data_iterator_iter)

        # shift tensors to GPU if available
        if params.cuda:
          levels = levels.cuda()
          children=children.cuda()
          n_inners=n_inners.cuda()
          contents=contents.cuda()
          n_level= n_level.cuda()
          labels_batch =labels_batch.cuda()
      
        # convert them to Variables to record operations in the computational graph
        levels=torch.autograd.Variable(levels)
        children=torch.autograd.Variable(children)
        n_inners=torch.autograd.Variable(n_inners)
        contents = torch.autograd.Variable(contents)
        n_level=torch.autograd.Variable(n_level)
        labels_batch = torch.autograd.Variable(labels_batch)    
    
        time_after_batch=time.time()
#         logging.info("Batch creation time" + str(time_after_batch-time_before_batch))
        
        ##-----------------------------
        # Feedforward pass through the NN
        output_batch = model(params, levels, children, n_inners, contents, n_level)
        
        
#         logging.info("Batch usage time" + str(time.time()-time_after_batch))
#         logging.info('####'*20)
        
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
    
    data_iterator_iter = iter(data_iterator)
    
    for _ in range(num_steps):
    
        # fetch the next evaluation batch
        levels, children, n_inners, contents, n_level, labels_batch=next(data_iterator_iter)

        # shift tensors to GPU if available
        if params.cuda:
          levels = levels.cuda()
          children=children.cuda()
          n_inners=n_inners.cuda()
          contents=contents.cuda()
          n_level= n_level.cuda()
          labels_batch =labels_batch.cuda()

        # convert them to Variables to record operations in the computational graph
        levels=torch.autograd.Variable(levels)
        children=torch.autograd.Variable(children)
        n_inners=torch.autograd.Variable(n_inners)
        contents = torch.autograd.Variable(contents)
        n_level=torch.autograd.Variable(n_level)
        labels_batch = torch.autograd.Variable(labels_batch)    

        ##-----------------------------
        # Feedforward pass through the NN
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

    ##------
    #Create lists to access the lenght below
    train_data=list(train_data)
    val_data=list(val_data)    
#     print('train data lenght=',len(train_data))
   
    num_steps_train=len(train_data)//params.batch_size
    num_steps_val=len(val_data)//params.batch_size
      
    # We truncate the dataset so that we get an integer number of batches    
    train_x=np.asarray([x for (x,y) in train_data][0:num_steps_train*params.batch_size])
    train_y=np.asarray([y for (x,y) in train_data][0:num_steps_train*params.batch_size])        
    val_x=np.asarray([x for (x,y) in val_data][0:num_steps_val*params.batch_size])
    val_y=np.asarray([y for (x,y) in val_data][0:num_steps_val*params.batch_size])
    
    ##------
    # Create tain and val datasets. Customized dataset class: dataset.TreeDataset that will create the batches by calling data_loader.batch_nyu_pad. 
    train_data = dataset.TreeDataset(data=train_x,labels=train_y,transform=data_loader.batch_nyu_pad,batch_size=params.batch_size,features=params.features)
    
    val_data = dataset.TreeDataset(data=val_x,labels=val_y,transform=data_loader.batch_nyu_pad,batch_size=params.batch_size,features=params.features,shuffle=False)
  
    ##------
    # Create the dataloader for the train and val sets (default Pytorch dataloader). Paralelize the batch generation with num_workers. BATCH SIZE SHOULD ALWAYS BE = 1 (batches are only loaded here as a single element, and they are created with dataset.TreeDataset).
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False,
                                               num_workers=8, pin_memory=True, collate_fn=dataset.customized_collate) 
                                               
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                               num_workers=8, pin_memory=True, collate_fn=dataset.customized_collate) 
    
    ##------
    # Train/evaluate for each epoch
    for epoch in range(params.num_epochs):
    
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
      
        # Train one epoch
        train(model, optimizer, loss_fn, train_loader, metrics, params, num_steps_train)
            
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, params, num_steps_val)      
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
#   nyu=False
  
  sample_name=''
  algo=''
  ##-------------------  
#   if nyu==True:
#     #Directory with the input trees
#     sample_name='nyu_jets'
#     
#   #   algo='antikt-antikt-delphes'
# #     algo='antikt-kt-delphes'
# #     algo='antikt-antikt'
#     algo=''
#     
#   else:
#     algo=''
#     
#     #Directory with the input trees
#     ### CHECK THAT SEARCH_HYPERPARAMS.PY HAS THE SAME SAMPLE NAME
#     
# #     sample_name='top_qcd_jets_antikt_antikt'
# #     sample_name='top_qcd_jets_antikt_kt'
#     sample_name='top_qcd_jets_antikt_CA'
    
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
  
  parser.add_argument('--jet_algorithm', default=algo, help="jet algorithm")
  parser.add_argument('--architecture', default='simpleRecNN', help="RecNN architecture")
  
  # Load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)

  ##-------------------
  # Set the logger
  utils.set_logger(os.path.join(args.model_dir, 'train.log'))
  
  dir_jets_subjets= args.data_dir
  algo=args.jet_algorithm
  
  architecture=args.architecture
  
  ##-------------------
  # Define file names with the trees of data. We rewrite the sample name if running from search_hyperparam.py
  sample_name=str(args.data_dir).split('/')[-1]
  logging.info('sample_name={}'.format(sample_name))
  logging.info('----'*20)
  
    # sample_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'+'_'+str(params.info)
  sample_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.info)
  
  logging.info('sample_filename={}'.format(sample_filename))
  
  train_data=data_dir+trees_dir+'train_'+sample_filename+'.pkl'
  val_data=data_dir+trees_dir+'dev_'+sample_filename+'.pkl'
  test_data=data_dir+trees_dir+'test_'+sample_filename+'.pkl'
    
  
  start_time = time.time()  
  
  


  ##----------------------------------------------------------------------------------------------------------
  ###   TRAINING
  ##----------------------------------------------------------------------------------------------------------
  data_loader=dl.DataLoader # Main class with the methods to load the raw data, create and preprocess the trees

  
  # use GPU if available
  params.cuda = torch.cuda.is_available()
  
  # Set the random seed for reproducible experiments
#   torch.manual_seed(230)
#   if params.cuda: torch.cuda.manual_seed(230)
  if params.cuda: torch.cuda.seed()
  ##-----------------------------
  # Create the input data pipeline 
  logging.info('---'*20)
  logging.info("Loading the datasets...")

  # Load data 
  with open(train_data, "rb") as f: train_data=pickle.load(f)
  with open(val_data, "rb") as f: val_data=pickle.load(f) 


  logging.info("- done loading the datasets") 
  logging.info('---'*20)   
  
  ##----------------------------------------------------------------------
  ## Architecture
  
  # Define the model and optimizer

  ## a) Simple RecNN 
  if architecture=='simpleRecNN': 
    model = net.PredictFromParticleEmbedding(params,make_embedding=net.GRNNTransformSimple).cuda() if params.cuda else net.PredictFromParticleEmbedding(params,make_embedding=net.GRNNTransformSimple) 
  
  ##----
  ## b) Gated RecNN
  elif architecture=='gatedRecNN':
    model = net.PredictFromParticleEmbeddingGated(params,make_embedding=net.GRNNTransformGated).cuda() if params.cuda else net.PredictFromParticleEmbeddingGated(params,make_embedding=net.GRNNTransformGated) 

  ## c) Leaves/inner different weights -  RecNN 
  if architecture=='leaves_inner_RecNN': 
    model = net.PredictFromParticleEmbeddingLeaves(params,make_embedding=net.GRNNTransformLeaves).cuda() if params.cuda else net.PredictFromParticleEmbeddingLeaves(params,make_embedding=net.GRNNTransformLeaves) 

  ##----
  ## d) Network in network (NiN) - Simple RecNN
  elif architecture=='NiNRecNN':
    model = net.PredictFromParticleEmbeddingNiN(params,make_embedding=net.GRNNTransformSimpleNiN).cuda() if params.cuda else net.PredictFromParticleEmbeddingNiN(params,make_embedding=net.GRNNTransformSimpleNiN)  

  ##----------------------------------------------------------------------
  # Output number of parameters of the model
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  pytorch_total_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  logging.info("Total parameters of the model= {}".format(pytorch_total_params))
  logging.info("Total weights of the model= {}".format(pytorch_total_weights))
  
  ##----------------------------------------------------------------------
  ## Optimizer and loss function
  
  logging.info("Model= {}".format(model))
  logging.info("---"*20)  
  logging.info("Building optimizer...")
  
  step_size=params.learning_rate
  decay=params.decay
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
    
    


    
    
    
  
