"""Aggregates results from the metrics_eval_best_weights.json and evaluate.log in a parent folder"""

from __future__ import print_function
import argparse
import json
import os
import numpy as np

from tabulate import tabulate



parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/simple_rnn_top_qcd',
                    help='Directory containing results of experiments')


#-------------------------------------------------------------------------------------------------------------
#/////////////////////    FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_val_best_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f) #The filename is the key and everything in the .json is the item
#             print('metrics[parent_dir]=',metrics[parent_dir])
       
    # Read the metrics from evaluate.log        
    evaluate_file = os.path.join(parent_dir, 'evaluate.log')
    if os.path.isfile(evaluate_file):
        with open(evaluate_file, 'r') as f:
          counter_auc=0
          counter_acc=0
          for line in f:
          
            # Read ROC auc          
            if line.strip().split()[-1].split('=')[0]=='roc_auc':
              counter_auc+=1 
              roc_auc_info=line.strip().split()[-1].split('=') 
#               print(str(roc_auc_info[0]),':',roc_auc_info[1])
              metrics[parent_dir][roc_auc_info[0]]=np.float(roc_auc_info[1])
#               print('metrics[parent_dir]=',metrics[parent_dir])

            # Read test accuracy
            if 'Eval metrics : accuracy:' in line:
              counter_acc+=1
#               print(line.strip().split())
              metrics[parent_dir]['test_accuracy']=np.float(line.strip().split()[7])
#               print('----'*20)   
#               print('metrics[parent_dir]=',metrics[parent_dir])

        #If there is no line matching the strings  
        if counter_auc==0:
          metrics[parent_dir]['roc_auc']=0
        if counter_acc==0:
          metrics[parent_dir]['test_accuracy']=0
    #Fill with zeros when there are missing files           
    else:
      if os.path.isfile(metrics_file):
        metrics[parent_dir]['roc_auc']=0
        metrics[parent_dir]['test_accuracy']=0
              

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
      if not os.path.isdir(os.path.join(parent_dir, subdir)):
        continue
      else:   
        aggregate_metrics(os.path.join(parent_dir, subdir), metrics)



  

#-------------------------------------------------------------------------------------------------------------
def metrics_to_table(metrics):
    ''' Generate a table with all the metrics for all the scans over the hyperparameter space and save a sorted list by ROC auc'''
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys() #keys of the json file
#     print('headers=',headers)
#     print('----'*20)
#     print('metrics.keys()=',metrics.keys())
#     print('----'*20)
#     print('metrics.values()=',metrics.values())
#     print('----'*20)
#     print('metrics.values()[h] for h in headers]=',[list(metrics.values())[0][h] for h in headers])
#     print('----'*20)
#     print('----'*20)

    # Make table
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()] #each value in values is a dictionary itself
    
    #Reorder elements and sort by ROC auc
    out_tuple=[(element[3],element[4],element[1],element[2],element[0][12::]) for element in table]
#     print('out_tuple=',out_tuple)
#     print('----'*20)
    dtype = [('roc_auc', float),('test_accuracy', float),('accuracy', float), ('loss', float),('name', np.unicode_,500)]
    sorted_results = np.sort(np.array(out_tuple, dtype=dtype),order='roc_auc')[::-1]
    
    return sorted_results

#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    args = parser.parse_args()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    sorted_results = metrics_to_table(metrics)
    
    # Display the table to terminal
    print('sorted_results=',sorted_results)
    print('----'*20)
    
    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, 'results_'+str(args.parent_dir).split('/')[-1]+'.dat')
    results=open(save_file,'w')      
    print('(test_roc_auc , test_accuracy , val_accuracy , val_loss , dir_name)',file=results)
    print(sorted_results,file=results)
        
        
        
        
        