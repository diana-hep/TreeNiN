"""Evaluates the model"""

import argparse
import logging
import os
import numpy as np
import torch
import sys
import time
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
# plt.ioff()
import pylab as pl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
from sklearn.utils import check_random_state

import model.data_loader as dl
from model import recNet as net
import utils
from model import preprocess 
import model.dataset as dataset

#-------------------------------------------------------------------------------------------------------------
#/////////////////////    EVALUATION FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
# Make ROC with area under the curve plot
def generate_results(y_test, y_score, params,weights=None):
    
    logging.info('length y_test={}'.format(len(y_test)))
    logging.info('Lenght y_score={}'.format(len(y_score)))
    
    if weights is not None: 
      logging.info('Sample weights length={}'.format(len(weights)))
    
      #We include the weights until the last full batch (the remaining ones are not enough to make a full batch)
      last_weight=len(weights)-len(weights)%params.batch_size
      weights=weights[0:last_weight]
      logging.info('New sample weights length={}'.format(len(weights)))
    
    ROC_plots_dir=args.model_dir+'/'
    
#     print('y_test=',y_test)
#     print('y_score=',y_score)
    
    # Get fpr, tpr
    if nyu==True:
      fpr, tpr, thresholds = roc_curve(y_test, y_score ,pos_label=1, sample_weight=weights, drop_intermediate=False)
    else:
      fpr, tpr, thresholds = roc_curve(y_test, y_score,pos_label=1, drop_intermediate=False)

    logging.info('Length y_score {}'.format(len(y_score)))
    logging.info('Length y_test {}'.format(len(y_test)))
    logging.info('Thresholds[0:6] = \n {}'.format(thresholds[:6]))
    logging.info('Thresholds lenght = \n{}'.format(len(thresholds)))
    logging.info('fpr lenght{}'.format(len(fpr)))
    logging.info('tpr lenght{}'.format(len(tpr)))
    if weights is not None: logging.info('Sample weights length={}'.format(len(weights)))
    if weights is not None: logging.info('Sample weights[0:4]={}'.format(weights[0:4]))
    
    # Save fpr, tpr to output file
#     rocnums=list(zip(fpr,tpr))
#     rocout=open(ROC_plots_dir+'roc_'+str(params.num_epochs)+'_'+sample_filename+'.csv','wb')
#     np.savetxt(rocout,rocnums,fmt="%10.5g",delimiter=',')

    # # Save fpr, tpr to output file
    with open(ROC_plots_dir+'roc_'+str(params.num_epochs)+'_'+sample_filename+'.pkl','wb') as f: pickle.dump(zip(fpr,tpr), f)
    

    logging.info('------------'*10)
    
    #Get ROC AUC
    if nyu==True:
      roc_auc = roc_auc_score(y_test, y_score, sample_weight=weights)
    else:
      roc_auc = roc_auc_score(y_test, y_score)  
      
    logging.info('roc_auc={}'.format(roc_auc))
    logging.info('------------'*10)


    #Plot ROC curve
    if plot_roc== True:
      plt.ioff()
      plt.figure()
  #     plt.ioff()
      plt.plot(tpr,1/fpr, color='red',label='Train epochs = '+str(params.num_epochs)+'\n ROC curve (area = %0.4f)' % roc_auc)
  #     plt.plot(fpr, tpr, color='red',label='Train epochs = '+str(params.num_epochs)+'\n ROC curve (area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xscale('log')
  #     plt.xlim([0.0, 1.05])
  #     plt.ylim([0.0, 1.05])
      plt.xlabel('Signal Tag Efficiency (tpr)')
      plt.ylabel('Background rejection (1/fpr)')
      plt.legend(loc="lower right")
      plt.title('Receiver operating characteristic curve')
      plt.grid(True)
  #     plt.show()
      fig = plt.gcf()
      label=''
      plot_FNAME = 'ROC_'+str(params.num_epochs)+'_'+sample_filename+'.png'
      plt.savefig(ROC_plots_dir+plot_FNAME)

      logging.info('AUC ={}'.format(np.float128(roc_auc)))
      logging.info('------------'*10)

    return roc_auc
    
#-------------------------------------------------------------------------------------------------------------
def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, sample_weights=None):
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
    out_prob=[]
    labels=np.array([])
    
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
        output_batch=output_batch.view((params.batch_size))
        loss = loss_fn(output_batch, labels_batch)
#         print('labels for loss=',labels_batch)
#         print('y_pred=',output_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        
#         print('concatenated labels before =', labels)
        labels=np.concatenate((labels,labels_batch))
#         print('concatenated labels after =', labels)
#         print('output_batch[0]=',output_batch[0:4])
#         print('output_batch shape=',output_batch.shape)

#         output_batch=output_batch.flatten()        
        out_prob=np.concatenate((out_prob,output_batch))
        
        # We calculate a single probability of tagging the image as signal        
#         for i_prob in range(len(output_batch)):
#             # out_prob.append((output_batch[i_prob][0]-output_batch[i_prob][1]+1)/2)
#             out_prob.append(output_batch[i_prob][1])

#         print('Predicted probability of each output neuron = \n',output_batch[0:15])
#         print('------------'*10)
#         print('Output of tagging image as signal = \n',np.array(out_prob)[-params.batch_size::])
#         print('------------'*10)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
     
    ##-----------------------------
    logging.info('Total Labels={}'.format(labels[0:10]))
    logging.info('Out prob={}'.format(out_prob[0:10]))
    logging.info('------------'*10)
    logging.info('len labels after ={}'.format(len(labels)))
    logging.info('len out_prob after{}'.format(len(out_prob)))
    
    # Get fpr, tpr, ROC curve and AUC
    roc_auc = generate_results(labels, out_prob, params,weights=sample_weights)
#     generate_results(labels, out_prob, params,weights=sample_weights)
    
    ## Save output prob and true values
    with open(out_files_dir+'yProbTrue_'+str(params.num_epochs)+'_'+sample_filename+'.pkl','wb') as f: pickle.dump(zip(out_prob, labels), f)
    
    
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    
    metrics_mean['auc']=roc_auc
    
    return metrics_mean

#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  """
      Evaluate the model on the test set.
  """
  ##-----------------------------------------------------------------------------------------------
  # Global variables
  ##-----------------
  data_dir='../data/'
  os.system('mkdir -p '+data_dir)

  plot_roc=False
  ##-----------------
  # Select the input sample 
#   nyu=True
  nyu=False
  
  
  ##-----------------  
  if nyu==True:
  
    #If true the batchization of the data is generated. Do it only once and the turn in off (only for nyu==True)
    make_batch=True
   #   make_batch=False
  
    #Directory with the input trees
    sample_name='nyu_jets'
    
  #   algo='antikt-antikt-delphes'
#     algo='antikt-kt-delphes'
#     algo='antikt-antikt'
    algo=''

  else:
    algo=''
    
    #Directory with the input trees
#     sample_name='top_qcd_jets_antikt_antikt'
#     sample_name='top_qcd_jets_antikt_kt'
    sample_name='top_qcd_jets_antikt_CA'
  
    #labels to look for the input files
  #   sg='tt'
    sg='ttbar' 
    bg='qcd'  
  
  
  ##---------------------------------------------------------------------------------------
    
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='../data/input_batches_pad/', help="Directory containing the input batches")
  parser.add_argument('--model_dir', default='experiments/base_model/', help="Directory containing params.json")
  parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                       containing weights to load")
    
  parser.add_argument('--trees_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
  parser.add_argument('--sample_name', default=sample_name, help="Sample name")
  parser.add_argument('--jet_algorithm', default=algo, help="jet algorithm")
  parser.add_argument('--architecture', default='simpleRecNN', help="RecNN architecture")
  
  # Load the parameters
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)
  
  out_files_dir=args.model_dir+'/'
  dir_jets_subjets= args.trees_dir
  algo=args.jet_algorithm
  
  architecture=args.architecture
  
  sample_name=args.sample_name #We rewrite the sample name when running from search_hyperparams.py
  
  print('sample_name=',sample_name)
  
#   sample_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'
  sample_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.info)
  logging.info('sample_filename={}'.format(sample_filename))
  ##-----------------
  # Get the logger
  utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

  # use GPU if available
  params.cuda = torch.cuda.is_available()     # use GPU is available

  # Set the random seed for reproducible experiments
#   torch.manual_seed(230)
#   if params.cuda: torch.cuda.manual_seed(230)
  if params.cuda: torch.cuda.seed()

  #---------------------------------------------------------------------------------------
  # Main class with the methods to load the raw data and create the batches 
  data_loader=dl.DataLoader 


  ##---------------------------------------------------------------------------------------
  ## Load batches of test data
  logging.info("Loading the dataset ...")
  test_data=args.data_dir+'test_'+sample_filename+'.pkl'
  
  if nyu==True:  
    with open(test_data, "rb") as f: test_data, test_weights =pickle.load(f)
  else:
    with open(test_data, "rb") as f: test_data =pickle.load(f)
  
  logging.info("- done.")



  ##-----------------------------------------------------------------------
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
  ## Loss function
  loss_fn = torch.nn.BCELoss()
#   loss_fn = torch.nn.CrossEntropyLoss()
  metrics = net.metrics  
  
  logging.info("Starting evaluation")

  # Reload weights from the saved file
  utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

  ##-----------------------------------------------------------------------------------------------
  # EVALUATE THE MODEL
  ##---------------------  
  test_data=list(test_data)
  num_steps_test=len(test_data)//params.batch_size
  
  print('num_steps_test=',num_steps_test)
           
  # We get an integer number of batches    
  test_x=np.asarray([x for (x,y) in test_data][0:num_steps_test*params.batch_size])
  test_y=np.asarray([y for (x,y) in test_data][0:num_steps_test*params.batch_size]) 
 
  ##------
  # Create tain and val datasets. Customized dataset class: dataset.TreeDataset that will create the batches by calling data_loader.batch_nyu_pad. 
  test_data = dataset.TreeDataset(data=test_x,labels=test_y,transform=data_loader.batch_nyu_pad,batch_size=params.batch_size,features=params.features,shuffle=False)

  ##------
  # Create the dataloader for the train and val sets (default Pytorch dataloader). Paralelize the batch generation with num_workers. BATCH SIZE SHOULD ALWAYS BE = 1 (batches are only loaded here as a single element, and they are created with dataset.TreeDataset).
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                               num_workers=8, pin_memory=True, collate_fn=dataset.customized_collate) 

  
  # Evaluate the model
  
  if nyu==True:
    test_metrics = evaluate(model, loss_fn, test_loader, metrics, params, num_steps_test, sample_weights=test_weights)
  else:
    test_metrics = evaluate(model, loss_fn, test_loader, metrics, params, num_steps_test)

    
  save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
  utils.save_dict_to_json(test_metrics, save_path)




















