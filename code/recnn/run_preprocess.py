"""Peform hyperparemeters search"""


# Comments
# info : This goes into the name of the batched dataset that we use to train/evaluate/test
# name: this goes into the name of the dir with the results (fpr/trp, evaluate, train log files, etc)

# DON'T CHANGE THE ARCHITECTURE, i.e. recNet.py for a run in between train and evaluate routines!!!
#-------------------------------------------------------------------------------------------------------------
import argparse
import os
from subprocess import check_call
import sys
import numpy as np
import utils
import time
#-------------------------------------------------------------------------------------------------------------
# Global variables
#-----------------------------

#Directory with the input trees
# sample_name='top_qcd_jets_antikt_antikt'
# sample_name='top_qcd_jets_antikt_kt'
# sample_name='top_qcd_jets_antikt_CA'

# New sample
# sample_name='top_qcd_jets_kt'
# sample_name='top_qcd_jets_kt_shift_rot_flip'

jet_algorithm=''

#------------------------------------------------------
#NYU samples
# sample_name='nyu_jets'
# jet_algorithm='antikt-antikt'


#Top tag reference dataset
sample_name='top_tag_reference_dataset'
jet_algorithm='kt'

#----------------
# architecture='gatedRecNN'
# architecture='simpleRecNN'
# architecture = 'leaves_inner_RecNN'
# architecture = 'NiNRecNN'
architecture = 'NiNRecNNReLU'
# architecture = 'NiNRecNN2L3W'
# architecture = 'NiNgatedRecNN'
#-------------------------------------------------------

# PREPROCESS=False
PREPROCESS=True

#-----------
# TRAIN_and_EVALUATE=True
TRAIN_and_EVALUATE=False

load_weights=False
# load_weights=True

#-----------
# EVALUATE=True
EVALUATE=False

# restore_file='last'
restore_file='best'

#-------------------------------------------------------
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=2,
                    help='Select the GPU')
parser.add_argument('--parent_dir', default='experiments/',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
parser.add_argument('--eval_data_dir', default='../data/preprocessed_trees/', help="Directory containing the input batches")
parser.add_argument('--sample_name', default=sample_name, help="Sample name")

parser.add_argument('--jet_algorithm', default=jet_algorithm, help="jet algorithm")

parser.add_argument('--architecture', default=architecture, help="RecNN architecture")


 
#-------------------------------------------------------------------------------------------------------------
#//////////////////////    FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
#------------------------------------------
# PREPROCESSING
def launch_preprocessing_job(parent_dir, data_dir, job_name, params,algo):

    start_time = time.time()
    print('search_hyperparams.py sample_name=',data_dir)
    print('----'*20)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Model dir=',model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)
        
    cmd_preprocess = "{python} preprocess_main.py --model_dir={model_dir} --data_dir={data_dir} --jet_algorithm={algo}".format(python=PYTHON, model_dir=model_dir, data_dir=data_dir, algo=algo)
    print(cmd_preprocess)
    check_call(cmd_preprocess, shell=True)

    elapsed_time=time.time()-start_time
    print('Preprocessing time (minutes) = ',elapsed_time/60)


#------------------------------------------
# TRAINING
def launch_training_job(parent_dir, data_dir, eval_data_dir, job_name, params, GPU,sample_name, algo):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    start_time = time.time()
    print('search_hyperparams.py sample_name=',data_dir)
    print('----'*20)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Model dir=',model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    
    if load_weights==False: 
      #---------------
      # Launch training with this config
      cmd_train = "CUDA_VISIBLE_DEVICES={gpu} {python} train.py --model_dir={model_dir} --data_dir={data_dir} --jet_algorithm={algo} --architecture={architecture}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=data_dir, algo=algo, architecture=architecture)
      print(cmd_train)
      check_call(cmd_train, shell=True)
    
    else:
      # Launch training with this config and restore previous weights(use --restore_file=best or --restore_file=last)
      cmd_train = "CUDA_VISIBLE_DEVICES={gpu} {python} train.py --model_dir={model_dir} --data_dir={data_dir}  --restore_file=best --jet_algorithm={algo} --architecture={architecture}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=data_dir, algo=algo, architecture=architecture)
      print(cmd_train)
      check_call(cmd_train, shell=True)


    elapsed_time=time.time()
    print('Training time (minutes) = ',(elapsed_time-start_time)/60)


#------------------------------------------
# EVALUATION
def launch_evaluation_job(parent_dir, data_dir, eval_data_dir, job_name, params, GPU,sample_name, algo):
    """Launch evaluation of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    elapsed_time = time.time()
    print('Running evaluation of the model')
    print('----'*20)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Model dir=',model_dir)


    #--------------
    # Launch evaluation with this config
    cmd_eval = "CUDA_VISIBLE_DEVICES={gpu} {python} evaluate.py --model_dir={model_dir} --data_dir={data_dir} --sample_name={sample_name} --jet_algorithm={algo} --architecture={architecture} --restore_file={restore_file}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=eval_data_dir,sample_name=sample_name, algo=algo, architecture=architecture, restore_file=restore_file)
    print(cmd_eval)
    check_call(cmd_eval, shell=True)

    eval_time=time.time()
    print('Evaluation time (minutes) = ',(eval_time-elapsed_time)/60)
    
    
#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'template_params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hyperparameters scans
    def multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[25],hidden_dims=[40],jet_numbers=[20000],Nfeatures=7,dir_name=None,name=None, info=None, sample_name=None, Nrun_start=0,Nrun_finish=1):
    
      parent_dir=args.parent_dir+str(dir_name)+'/'
      if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
#           os.system('mkdir -p '+parent_dir)

      #-------------------------------------------------------------
      # Loop to scan over the hyperparameter space
      for jet_number in jet_numbers:
        for hidden_dim in hidden_dims:
          for num_epoch in num_epochs:
            for batch_size in batch_sizes:
              for decay in decays: 
                for learning_rate in learning_rates:

                  # Modify the relevant parameter in params
                  params.learning_rate=0.002
                  params.decay=0.9
                  params.batch_size=128
                  params.num_epochs=25
                  params.save_summary_steps=params.batch_size
                  params.hidden=40
                  params.features=7
                  params.number_of_labels_types=1
                  params.myN_jets=20000

                  params.learning_rate=learning_rate
                  params.decay=decay
                  params.batch_size=batch_size
                  params.num_epochs=num_epoch
                  params.hidden=hidden_dim
                  params.features=Nfeatures
        #           params.number_of_labels_types=1
                  params.myN_jets=jet_number
                  params.info=info #This goes into the name of the batched dataset that we use to train/evaluate/test
                  params.nrun_start=Nrun_start
                  params.nrun_finish=Nrun_finish
                  #-----------------------------------------
                  # Launch job (name has to be unique)
                  job_name = str(sample_name)+'_'+str(name)+'_lr_'+str(learning_rate)+'_decay_'+str(decay)+'_batch_'+str(batch_size)+'_epochs_'+str(num_epoch)+'_hidden_'+str(hidden_dim)+'_Njets_'+str(jet_number)+'_features_'+str(params.features)
                  
                  
                  
                  #-----------------------------------------                
                  # Run preprocess, training, evaluation 
                  if PREPROCESS:
                    launch_preprocessing_job(parent_dir, args.data_dir, job_name, params, jet_algorithm)
                  
                  if TRAIN_and_EVALUATE:
                    for n_run in np.arange(Nrun_start,Nrun_finish):
                      launch_training_job(parent_dir, args.data_dir, args.eval_data_dir, job_name+'/run_'+str(n_run), params, args.gpu, sample_name, jet_algorithm)        

                      launch_evaluation_job(parent_dir, args.data_dir, args.eval_data_dir, job_name+'/run_'+str(n_run), params, args.gpu, sample_name, jet_algorithm)


                  if EVALUATE:
                    for n_run in np.arange(Nrun_start,Nrun_finish):
                      launch_evaluation_job(parent_dir, args.data_dir, args.eval_data_dir, job_name+'/run_'+str(n_run), params, args.gpu, sample_name, jet_algorithm) 

    #---------------------------------------------------------------------------------------------------------
    # SCANS OVER THE HYPERPARAMETER SPACE        
    #-------------------
    ##TESTS

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[8],num_epochs=[1],hidden_dims=[50],jet_numbers=[20], Nfeatures=7,dir_name='networkingNet_rnn_top_qcd/kt',name='kt_1layer_4weights', info='kt_1layer_4weights',sample_name=args.sample_name) #gpu0  s  
   
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[5],hidden_dims=[50], jet_numbers=[5000], Nfeatures=7,dir_name='kt/NiN_rnn_top_qcd',name='test', info='permute_nleaves',sample_name=args.sample_name,Nrun_start=1,Nrun_finish=10) #gpu1
   
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[1],hidden_dims=[50],jet_numbers=[3000], Nfeatures=7,dir_name='networkingNet_rnn_top_qcd/time_test',name='test', info='',sample_name=args.sample_name) #gpu1
   
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[2],num_epochs=[1],hidden_dims=[50], jet_numbers=[10], Nfeatures=7,dir_name='nyu',name='test', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=1) #gpu1 
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    
    #---------------------------------------------------------------------
    # Full dataset scans   

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50],jet_numbers=[600000], Nfeatures=7,dir_name='networkingNet_rnn_top_qcd/kt_dec13',name='2layer_3W_inner_outer_NoReLU_NoRoot_orthogonal_uk', info='',sample_name=args.sample_name) #gpu0


#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50],jet_numbers=[100000], Nfeatures=7,dir_name='networkingNet_rnn_top_qcd/kt_feb1',name='2layer_3W_inner_outer_NoRoot_orthogonal_uk', info='',sample_name=args.sample_name) #gpu1

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50],jet_numbers=[100000], Nfeatures=7,dir_name='networkingNet_rnn_top_qcd/kt_feb1',name='uk_inner_outer_2layer_3W_inner_outer_NoRoot_orthogonal', info='',sample_name=args.sample_name) #gpu1

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50],jet_numbers=[600000], Nfeatures=7,dir_name='gated_rnn_top_qcd/kt',name='kt', info='kt',sample_name=args.sample_name) #gpu2 


#-------------------------------------------------------------------------------------------------
# NYU samples - validation of the code

#Simple RecNN- antikt - particles
#     multi_scan(learning_rates=[5e-4],decays=[0.9], batch_sizes=[64],num_epochs=[25],hidden_dims=[40], jet_numbers=[100000], Nfeatures=7,dir_name='nyu_jet',name='antikt-antikt', info='',sample_name=args.sample_name,Nrun_start=25,Nrun_finish=30) #gpu1 

#Gated RecNN -antikt - particles
#     multi_scan(learning_rates=[5e-4],decays=[0.9], batch_sizes=[64],num_epochs=[25],hidden_dims=[40], jet_numbers=[100000], Nfeatures=7,dir_name='nyu_jet',name=architecture+'_antikt-antikt', info='',sample_name=args.sample_name,Nrun_start=29,Nrun_finish=30) #gpu1 

#-------------------------------------------------------------------------------------------------

#Gated RecNN -antikt - particles
#     multi_scan(learning_rates=[5e-4],decays=[0.9], batch_sizes=[64],num_epochs=[25],hidden_dims=[40], jet_numbers=[100000], Nfeatures=7,dir_name='test_nworker15',name=architecture+'_antikt-antikt', info='',sample_name=args.sample_name,Nrun_start=2,Nrun_finish=3) #gpu1 


# Simple (run 0-10)/Gated RecNN - Top tag Reference Dataset. (Don't run more than 3 at the same time)
#     multi_scan(learning_rates=[5e-4],decays=[0.9], batch_sizes=[64],num_epochs=[25],hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt', info='',sample_name=args.sample_name,Nrun_start=7,Nrun_finish=10) #gpu1 

# Simple (run 0-10)/Gated RecNN - Top tag Reference Dataset. (Don't run more than 3 at the same time)
#     multi_scan(learning_rates=[5e-4],decays=[0.9], batch_sizes=[64],num_epochs=[30],hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt', info='',sample_name=args.sample_name,Nrun_start=7,Nrun_finish=10) #gpu1 

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt', info='',sample_name=args.sample_name,Nrun_start=6,Nrun_finish=9) #gpu1 


#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=6,Nrun_finish=9) #gpu1


#     multi_scan(learning_rates=[0.0000295],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk_80epochs', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=3) #gpu1

#     multi_scan(learning_rates=[0.0000295],decays=[0.86], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk_80epochs', info='',sample_name=args.sample_name,Nrun_start=1,Nrun_finish=3) #gpu1

#     multi_scan(learning_rates=[2e-3],decays=[0.86], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L3WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=1) #gpu2

#     multi_scan(learning_rates=[2e-3],decays=[0.86], batch_sizes=[128],num_epochs=[45],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L3WleavesInnerNiNuk_loss', info='',sample_name=args.sample_name,Nrun_start=6,Nrun_finish=9) #gpu2


#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L6WleavesInnerNiN_ukh', info='',sample_name=args.sample_name,Nrun_start=3,Nrun_finish=6) #gpu1

#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L6WleavesInnerNiN_ukh', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=3) #gpu1


#     multi_scan(learning_rates=[2e-2],decays=[0.845], batch_sizes=[128],num_epochs=[45],hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=3) #gpu1


#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk_tanh', info='',sample_name=args.sample_name,Nrun_start=3,Nrun_finish=6) #gpu1


#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=9,Nrun_finish=12) #gpu1

#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[50],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk_bgRejectionbest', info='',sample_name=args.sample_name,Nrun_start=15,Nrun_finish=18) #gpu1

#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[50],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk_bgRejectionbest', info='',sample_name=args.sample_name,Nrun_start=6,Nrun_finish=9) #gpu1


#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=6,Nrun_finish=9) #gpu1

##########################################################
# Best performing model with architecture = 'NiNRecNN'. Note: This is the same architecture as NiNRecNNReLU but here we initialize some weights that are not used. We need to load this one for testing even thought there are extra weights (Because the saved weight include these extra weights).

# top_tag_reference_dataset_NiNRecNN_kt_full_test_2L4WleavesInnerNiNuk_lr_0.002_decay_09_batch_400_epochs_40_hidden_50_Njets_1200000_features_7
    multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[400],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_last_kt_full_test_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=1) #gpu1   


#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[400],num_epochs=[40],hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_full_test_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=1) #gpu1 
    
###########################################################

#-------------------------------
# NYU PREPROCESSING - rot_boost_rot_flip

#Simple
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128], num_epochs=[40], hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7, dir_name='top_tag_reference_dataset', name=architecture+'_kt_R_0.3_rot_boost_rot_flip', info='R_0.3_rot_boost_rot_flip', sample_name=args.sample_name, Nrun_start=6, Nrun_finish=9) #gpu1 

#-------------
# NiNRelu
#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128], num_epochs=[45], hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7, dir_name='top_tag_reference_dataset', name=architecture+'_kt_R_0.3_rot_boost_rot_flip', info='R_0.3_rot_boost_rot_flip', sample_name=args.sample_name, Nrun_start=6, Nrun_finish=9) #gpu1 

# Evaluate on last weights
#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128], num_epochs=[45], hidden_dims=[50], jet_numbers=[1200000], Nfeatures=7, dir_name='top_tag_reference_dataset', name=architecture+'_kt_R_0.3_rot_boost_rot_flip_last', info='R_0.3_rot_boost_rot_flip', sample_name=args.sample_name, Nrun_start=7, Nrun_finish=9) #gpu1 

#     multi_scan(learning_rates=[5e-4],decays=[0.92], batch_sizes=[64], num_epochs=[40], hidden_dims=[40], jet_numbers=[1200000], Nfeatures=7, dir_name='top_tag_reference_dataset', name=architecture+'_kt_R_0.3_rot_boost_rot_flip', info='R_0.3_rot_boost_rot_flip', sample_name=args.sample_name, Nrun_start=6, Nrun_finish=9) #gpu1























#----------------------------------------------------------
# Smaller dataset - tests NiN, etc. We have 120k total training, 40k val and 40 test
#     multi_scan(learning_rates=[5e-4],decays=[0.9], batch_sizes=[64],num_epochs=[10],hidden_dims=[40], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_test', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=1) #gpu1 

# Smaller dataset - tests NiN, etc. We have 120k total training, 40k val and 40 test
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L3WleavesInnerNiN', info='',sample_name=args.sample_name,Nrun_start=9,Nrun_finish=12) #gpu1 

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiN', info='',sample_name=args.sample_name,Nrun_start=9,Nrun_finish=12) #gpu1 


#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=4) #gpu1 

# Smaller dataset - tests NiN, etc. We have 120k total training, 40k val and 40 test
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L3WleavesInnerNiN_tanh', info='',sample_name=args.sample_name,Nrun_start=9,Nrun_finish=10) #gpu1 

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_2L4WleavesInnerNiNuk', info='',sample_name=args.sample_name,Nrun_start=10,Nrun_finish=11) #gpu1 


#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[5],hidden_dims=[50], jet_numbers=[120000], Nfeatures=7,dir_name='top_tag_reference_dataset',name=architecture+'_kt_test', info='',sample_name=args.sample_name,Nrun_start=0,Nrun_finish=1) #gpu1 








'''
antikt_antikt
learning_rates=[1e-2, 5e-3,2e-3,5e-4] 
decays=[0.9,0.8,0.7]
batch_sizes=[64,128,256,512,1024]
num_epochs=[30]
hidden_dims=[20,40,80,160,320,640]
jet_numbers = [40000,80000,160000]

-----------------------------------
antikt_kt
learning_rates=[1e-2, 5e-3,2e-3,1e-3] 
decays=[0.9,0.8,0.7]
batch_sizes=[64,128,256,512,1024]
num_epochs=[35]
hidden_dims=[20,40,80,160,320,640]
jet_numbers = [40000,80000,160000]
'''       
      
        