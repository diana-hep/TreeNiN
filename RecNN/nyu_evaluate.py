"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
# from model.data_loader import DataLoader

import sys

import model.data_loader as dl
from model import recNet as net
# from model.data_utils import tree_list as tree_class

import time
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
from sklearn.utils import check_random_state

from model import preprocess 
import model.data_loader as dl
#-------------------------------------------------------------------------------------------------------------
# Make ROC with area under the curve plot
def generate_results(y_test, y_score,weights, params):
    
    logging.info('length y_test={}'.format(len(y_test)))
    logging.info('Lenght y_score={}'.format(len(y_score)))
    logging.info('Sample weights length={}'.format(len(weights)))
    
    #We include the weights until the last full batch. The remaining ones are not enough to make a batch
    last_weight=len(weights)-len(weights)%params.batch_size
    weights=weights[0:last_weight]
    logging.info('Sample weights length after={}'.format(len(weights)))
    
    ROC_plots_dir=args.model_dir+'/'
    #I modified from pos_label=1 to pos_label=0 because I found out that in my code signal is labeled as 0 and bg as 1

    fpr, tpr, thresholds = roc_curve(y_test, y_score ,pos_label=1, sample_weight=weights, drop_intermediate=False)
#     fpr, tpr, thresholds = roc_curve(y_test, y_score,pos_label=1, drop_intermediate=False)

    logging.info('Length y_score {}'.format(len(y_score)))
    logging.info('Length y_test {}'.format(len(y_test)))
    logging.info('Thresholds[0:6] = \n {}'.format(thresholds[:6]))
    logging.info('Thresholds lenght = \n{}'.format(len(thresholds)))
    logging.info('fpr lenght{}'.format(len(fpr)))
    logging.info('tpr lenght{}'.format(len(tpr)))
    logging.info('Sample weights length={}'.format(len(weights)))
    logging.info('Sample weights[0:4]={}'.format(weights[0:4]))
    
    rocnums=list(zip(fpr,tpr))
    rocout=open(ROC_plots_dir+'roc_'+str(params.num_epochs)+'_'+batch_filename+'.csv','wb')
    np.savetxt(rocout,rocnums,fmt="%10.5g",delimiter=',')


    logging.info('------------'*10)
    roc_auc = roc_auc_score(y_test, y_score, sample_weight=weights)
    logging.info('roc_auc={}'.format(roc_auc))

#     logging.info('------------'*10)
#     roc_auc = roc_auc_score(y_test, y_score)
#     logging.info('roc_auc={}'.format(roc_auc))

    
    plt.figure()
    plt.plot(tpr,1/fpr, color='red',label='Train epochs = '+str(params.num_epochs)+'\n ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot(fpr, tpr, color='red',label='Train epochs = '+str(params.num_epochs)+'\n ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot(fpr[2], tpr[2], color='red',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
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
    plot_FNAME = 'ROC_'+str(params.num_epochs)+'_'+batch_filename+'.png'
    plt.savefig(ROC_plots_dir+plot_FNAME)

 #   ROC_FNAME = 'ROC_'+str(epochs)+'_'+in_tuple+label+'_Ntrain_'+str(Ntrain)+'.npy'
 #   np.save(ROC_plots_dir+'fpr_'+str(sample_relative_size)+'_'+ROC_FNAME,fpr)
 #   np.save(ROC_plots_dir+'tpr_'+str(sample_relative_size)+'_'+ROC_FNAME,tpr)
 #   print('ROC filename = {}'.format(ROC_plots_dir+plot_FNAME))
    logging.info('AUC ={}'.format(np.float128(roc_auc)))
    logging.info('------------'*10)


#print('Predicting on test data')
#y_score = model.predict(X_test)

#print('Generating results')
#print('y_Pred[:,0]',y_Pred[:, 0])

#print('Y_Pred',Y_Pred[0:15])
#Y_Pred_prob = model.predict_proba(x_test)
#print('Y_Pred_prob',Y_Pred_prob[0:15])









def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, sample_weights):
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
#     for i_batch in range(1):
        # fetch the next evaluation batch
#         isleaf, left, right, contents, labels_batch  = next(data_iterator)

        if pad==True:
          levels, children, n_inners, contents, n_level, labels_batch = next(data_iterator)
          output_batch = model(params, levels, children, n_inners, contents, n_level)
        else:
          levels, children, n_inners, contents, labels_batch = next(data_iterator)
          output_batch = model(params, levels, children, n_inners, contents)
          
        
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
#         
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
#         summary_batch['loss'] = loss.data[0]
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

        
    ##-----------------------------
    print('Labels=',labels[0:10])
    print('Out prob=',out_prob[0:10])
    
    print('len labels after =',len(labels))
    print('len out_prob after',len(out_prob))
    
    # Get fpr, tpr and AUC
    generate_results(labels, out_prob,sample_weights, params)
    
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
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
  
  #labels to look for the input files
#   sg='tt'
  sg='ttbar' 
  bg='qcd'
  
  #Directory with the input trees
#   sample_name='top_qcd_jets'
#   sample_name='test_anti_kt'
#   sample_name='kt_test'
  sample_name='nyu_jets' 
  
#   algo='antikt-antikt-delphes'
  #   algo='' 
#   algo='antikt-kt-delphes'
  algo='antikt-antikt'
  
  file_name=algo+'-test.pickle'

  
  make_batch=True
#   make_batch=False
  nyu=True
#   nyu=False
  ##-------------------------------
    
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='../data/input_batches_pad/', help="Directory containing the input batches")
  parser.add_argument('--model_dir', default='experiments/base_model/', help="Directory containing params.json")
  parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                       containing weights to load")
    
  parser.add_argument('--trees_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
  
  
  # Load the parameters
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)

  dir_jets_subjets= args.trees_dir
  batch_filename=sample_name+'_'+algo+'_'+str(params.myN_jets)+'_Njets_'+str(params.batch_size)+'_batch'

  # use GPU if available
  params.cuda = torch.cuda.is_available()     # use GPU is available

  # Set the random seed for reproducible experiments
  torch.manual_seed(230)
  if params.cuda: torch.cuda.manual_seed(230)
   
   
  pad=params.pad 
      
  # Get the logger
  utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

  # Create the input data pipeline
  logging.info("Creating the dataset...")
#---------------------------------------------------------------------------------------
  # Create the batches (do it only once, then turn make_batch=False)
  if make_batch==True:
    
    if nyu==True: # We load the nyu samples
      
      batches_dir='input_batches_pad/'
      test_data=data_dir+batches_dir+'test_'+batch_filename+'.pkl'
      data_loader=dl.DataLoader # Main class with the methods to load the raw data and create the batches

      
      nyu_train=dir_jets_subjets+'/'+file_name
      # loading dataset_params and make trees
      print('nyu_train=',nyu_train)
      fd = open(nyu_train, "rb")
      X, y = pickle.load(fd,encoding='latin-1')
      fd.close()
      
#       with open(nyu_train, "rb") as f: X,Y=pickle.load(f)
      
      X=np.asarray(X)
      y=np.asarray(y)
      
      print('Training data size=',len(X))
      print('---'*20)
#       print('X[0]=',np.asarray(X[0]))
#       print('----'*20)
#       print('y[0]=',np.asarray(Y[0]))
#       print('----'*20)
#       sys.exit()
      
      # Shuffle the sets
      indices = check_random_state(1).permutation(len(X))
      X = X[indices]
      y = y[indices]  
      X=np.asarray(X)
      y=np.asarray(y)
      
      #Ensure that the left sub-jet has always a larger pt than the right. Change the input variables
      X = [preprocess.extract(preprocess.permute_by_pt(preprocess.rewrite_content(jet))) for jet in X]
#       
#       #Ensure that the left sub-jet has always a larger pt than the right
#       X=[preprocess.permute_by_pt(jet) for jet in X]
#       
#       # Change the input variables
#       X=[preprocess.extract(jet) for jet in X]
      
      
      X=data_loader.scale_features(X) 

    
      # Cropping
      X_ = [j for j in X if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]
      y_ = [y[i] for i, j in enumerate(X) if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]

      X = X_
      y = y_
      X=np.asarray(X)
      y = np.asarray(y)
    
      print('Lenght X=', len(X))
      print('Length y=', len(y))
      
      
      
      # Weights for flatness in pt
      w = np.zeros(len(y))    
    
      print('Length w before=', len(w))
      
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

      print('Length w after=', len(w))

      test_batches=dl.batch_array(X, y, params.batch_size, params.features)
      print('Number of test_batches=',len(test_batches))  


      with open(test_data, "wb") as f: pickle.dump((test_batches,w), f)
#       sys.exit()
#---------------------------------------------------------------------------------------
      
  # load data
#   data_dir='../data/'
#   batches_dir='input_batches_pad/'


  # test_data=data_dir+batches_dir+'test_'+batch_filename+'.pkl'
  test_data=args.data_dir+'test_'+batch_filename+'.pkl'
#   test_data=args.data_dir+'dev_'+batch_filename+'.pkl'
  
  with open(test_data, "rb") as f: test_data, test_weights =pickle.load(f)
  
#   data_loader = DataLoader(args.data_dir, params)
#   data = data_loader.load_data(['test'], args.data_dir)
#   test_data = data['test']
# 
#   # specify the test set size
#   params.test_size = test_data['size']
#   test_data_iterator = data_loader.data_iterator(test_data, params)

  logging.info("- done.")



  # Define the model
  model = net.GRNNTransformSimple_level(params).cuda() if params.cuda else net.GRNNTransformSimple_level(params)   

  loss_fn = torch.nn.BCELoss()
#   loss_fn = torch.nn.CrossEntropyLoss()
  metrics = net.metrics  

  
  logging.info("Starting evaluation")

  # Reload weights from the saved file
  utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

  # Evaluate
  num_steps_test=len(test_data)
#   num_steps = (params.test_size + 1) // params.batch_size

  ##-----------------------------------------------------------------------------------------------
  # Evaluate for one epoch on validation set
  data_loader=dl.DataLoader # Main class with the methods to load the raw data and create the batches 
  test_data_iterator = data_loader.make_pad_batch_iterator_level(test_data, params.batch_size)
  test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps_test, test_weights)

#   test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
  save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
  utils.save_dict_to_json(test_metrics, save_path)




















