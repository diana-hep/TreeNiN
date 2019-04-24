import sys, os
# sys.path.append("..")

import numpy as np
np.seterr(divide="ignore")
import logging
import pickle
import scipy as sp
import json


# from sklearn.utils import check_random_state
# from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from scipy import interp

#--------------------------------------------------------------------------
# Load all the info for each run in a dictionary (hyperparameteres, auc, fpr, tpr, output prob, etc)
def load_results(hyperparam_dir,show_data=False):  
    results_dictionary = []
#     for folder in os.listdir(hyperparam_dir):
#         print(folder)
    names = hyperparam_dir.split('/')[-1].split("_")
    if show_data:
        print('names=',names)
        
    Name=names[-15]
    lr = names[-13]
    decay = names[-11]
    batch = names[-9]
    epochs = names[-7]
    hidden = names[-5]
    Njets = names[-3]
    Nfeatures = names [-1]
    for subdir in os.listdir(hyperparam_dir):
        if subdir.startswith('run') and ('metrics_test_best.json' in os.listdir(hyperparam_dir+'/'+subdir)):
#             print(subdir)
            with open(hyperparam_dir+'/'+subdir+'/metrics_test_best.json') as f:
                data = json.load(f)
#             rocname = [filename for filename in np.sort(os.listdir(hyperparam_dir+'/'+subdir)) if filename.startswith('roc_')][0]
            outprobname =[filename for filename in np.sort(os.listdir(hyperparam_dir+'/'+subdir)) if filename.startswith('yProbTrue_')][0]
#             with open(hyperparam_dir+'/'+subdir+'/'+rocname, "rb") as f: roc=list(pickle.load(f))
            with open(hyperparam_dir+'/'+subdir+'/'+outprobname, "rb") as f: yProbTrue=list(pickle.load(f))
    #             print(list(roc))
    #             fpr=[x for (x,y) in roc]
    #             tpr=[y for (x,y) in roc]
    #             print('fpr = ',fpr)
            dictionary = {'name':Name,
                          'runName':subdir,
                          'lr':float(lr),
                          'decay':float(decay),
                          'batch':int(batch),
                          'hidden':int(hidden),
                          'Njets':Njets,
                          'Nfeatures':Nfeatures,
                          'accuracy':data['accuracy'], 
                          'loss':data['loss'],
                          'auc':data['auc'],
#                           'roc':np.asarray(roc),
#                           'fpr':np.asarray([x for (x,y) in roc]),
#                           'tpr':np.asarray([y for (x,y) in roc]),
                          'yProbTrue':np.asarray(yProbTrue)}

            results_dictionary.append(dictionary)
            if show_data:
                 print(dictionary)


    return results_dictionary
    
    
#-------------
def save_out_prob(yProbTrue, out_prob_dir, input_name=''):
    
#     for i in range(np.shape(yProbTrue)[0]):
    #   print('yProbTrue shape=',np.shape(yProbTrue[i]))
#             print('yProbTrue=',yProbTrue[top_ref_run][0:5])
    print('yProb shape=',np.shape(yProbTrue[:,:,0]))
    print('Saving output probabilities and true values here: '+out_prob_dir+str(input_name)+'.pkl')
    with open(out_prob_dir+str(input_name)+'.pkl', "wb") as f: pickle.dump(yProbTrue[:,:,0], f)
    
##=============================================================================    
    
# Full test dataset of  404k jets - hd=50
results_dir='recnn/experiments/top_tag_reference_dataset/top_tag_reference_dataset_NiNRecNNReLU_last_kt_full_test_2L4WleavesInnerNiNuk_lr_0.002_decay_0.9_batch_400_epochs_40_hidden_50_Njets_1200000_features_7'

prob_dir= 'top_reference_dataset/best_models/'
os.system('mkdir -p '+prob_dir)

results_dic_best= load_results(results_dir, show_data=False)

YProbBest=np.asarray([element['yProbTrue'] for element in results_dic_best])

save_out_prob(YProbBest, prob_dir, input_name='TreeNiN_hd50')




