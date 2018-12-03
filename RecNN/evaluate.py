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
    rocnums=list(zip(fpr,tpr))
    rocout=open(ROC_plots_dir+'roc_'+str(params.num_epochs)+'_'+batch_filename+'.csv','wb')
    np.savetxt(rocout,rocnums,fmt="%10.5g",delimiter=',')

    logging.info('------------'*10)
    
    #Get ROC AUC
    if nyu==True:
      roc_auc = roc_auc_score(y_test, y_score, sample_weight=weights)
    else:
      roc_auc = roc_auc_score(y_test, y_score)  
      
    logging.info('roc_auc={}'.format(roc_auc))
    logging.info('------------'*10)
# 
#     #Plot ROC curve
#     plt.ioff()
#     plt.figure()
# #     plt.ioff()
#     plt.plot(tpr,1/fpr, color='red',label='Train epochs = '+str(params.num_epochs)+'\n ROC curve (area = %0.4f)' % roc_auc)
# #     plt.plot(fpr, tpr, color='red',label='Train epochs = '+str(params.num_epochs)+'\n ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xscale('log')
# #     plt.xlim([0.0, 1.05])
# #     plt.ylim([0.0, 1.05])
#     plt.xlabel('Signal Tag Efficiency (tpr)')
#     plt.ylabel('Background rejection (1/fpr)')
#     plt.legend(loc="lower right")
#     plt.title('Receiver operating characteristic curve')
#     plt.grid(True)
# #     plt.show()
#     fig = plt.gcf()
#     label=''
#     plot_FNAME = 'ROC_'+str(params.num_epochs)+'_'+batch_filename+'.png'
#     plt.savefig(ROC_plots_dir+plot_FNAME)

#     logging.info('AUC ={}'.format(np.float128(roc_auc)))
#     logging.info('------------'*10)

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
    for i_batch in range(num_steps):
       
        levels, children, n_inners, contents, n_level, labels_batch = next(data_iterator)
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

        labels=np.concatenate((labels,labels_batch))
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
    logging.info('Labels={}'.format(labels[0:10]))
    logging.info('Out prob={}'.format(out_prob[0:10]))
    logging.info('------------'*10)
    logging.info('len labels after ={}'.format(len(labels)))
    logging.info('len out_prob after{}'.format(len(out_prob)))
    
    # Get fpr, tpr, ROC curve and AUC
    generate_results(labels, out_prob, params,weights=sample_weights)
    
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
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
    algo='antikt-antikt'
  
    file_name=algo+'-test.pickle'
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
  
  # Load the parameters
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)

  dir_jets_subjets= args.trees_dir
  sample_name=args.sample_name #We rewrite the sample name when running from search_hyperparams.py
  
  print('sample_name=',sample_name)
  
  batch_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'

  ##-----------------
  # Get the logger
  utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

  # use GPU if available
  params.cuda = torch.cuda.is_available()     # use GPU is available

  # Set the random seed for reproducible experiments
  torch.manual_seed(230)
  if params.cuda: torch.cuda.manual_seed(230)

  #---------------------------------------------------------------------------------------
  # Main class with the methods to load the raw data and create the batches 
  data_loader=dl.DataLoader 
  
  # Create the batches (do it only once, then turn make_batch=False)
  if nyu==True: # We load the nyu samples
  
    # Create the input data pipeline
    logging.info("Creating the dataset...")
    
    if make_batch==True:
      
      batches_dir='input_batches_pad/'
      test_data=data_dir+batches_dir+'test_'+batch_filename+'.pkl'
      
  
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
      X = [preprocess.extract(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in X]


      # Apply RobustScaler (remove outliers, center and scale data) with train set transformer
      transformer_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'
      transformer_data=data_dir+batches_dir+'transformer_'+transformer_filename+'.pkl'      
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

      #------------------
      # Generate dataset batches.
      test_batches=dl.batch_array(X, y, params.batch_size, params.features)
      logging.info('Number of test_batches='+str(len(test_batches)))  

      # Save batches
      with open(test_data, "wb") as f: pickle.dump((test_batches,w), f)
      
      logging.info("- done.")
      #---------------------------------------------------------------------------------------

  # Load batches of test data
  logging.info("Loading the dataset ...")
#   test_data=args.data_dir+'test_'+batch_filename+'7_features'+'.pkl'
#   test_data=args.data_dir+'test_'+batch_filename+'.pkl'
  test_data=args.data_dir+'test_'+batch_filename+'_'+str(params.info)+'.pkl'
  
  if nyu==True:  
    with open(test_data, "rb") as f: test_data, test_weights =pickle.load(f)
  else:
    with open(test_data, "rb") as f: test_data =pickle.load(f)
  
  logging.info("- done.")

  #-------------------------
  # Define the model
  model = net.GRNNTransformSimple_level(params).cuda() if params.cuda else net.GRNNTransformSimple_level(params)   

  #Loss function
  loss_fn = torch.nn.BCELoss()
#   loss_fn = torch.nn.CrossEntropyLoss()
  metrics = net.metrics  

  #-------------------------
  logging.info("Starting evaluation")

  # Reload weights from the saved file
  utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)


  ##-----------------------------------------------------------------------------------------------
  # Evaluate the model
  ##---------------------
  
  # Run the data iterator
  test_data_iterator = data_loader.make_pad_batch_iterator_level(test_data, params.batch_size)
  
  # Evaluate the model
  num_steps_test=len(test_data)
  
  if nyu==True:
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps_test, sample_weights=test_weights)
  else:
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps_test)
    
    
  save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
  utils.save_dict_to_json(test_metrics, save_path)




















