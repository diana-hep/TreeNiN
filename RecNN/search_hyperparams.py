"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils

#-------------------------------------------------------------------------------------------------------------
# Global variables
#-----------------------------

#Directory with the input trees
# sample_name='top_qcd_jets_antikt_antikt'
sample_name='top_qcd_jets_antikt_kt'
# sample_name='top_qcd_jets_antikt_CA'


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=2,
                    help='Select the GPU')
parser.add_argument('--parent_dir', default='experiments/',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='../data/inputTrees/'+sample_name, help="Directory containing the raw datasets")
parser.add_argument('--eval_data_dir', default='../data/input_batches_pad/', help="Directory containing the input batches")

#-------------------------------------------------------------------------------------------------------------
#/////////////////////    FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------
def launch_training_job(parent_dir, data_dir, eval_data_dir, job_name, params, GPU):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    
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
    
    # Launch training with this config
    cmd_train = "CUDA_VISIBLE_DEVICES={gpu} {python} train.py --model_dir={model_dir} --data_dir={data_dir}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd_train)
    check_call(cmd_train, shell=True)
    

#     # Launch training with this config and restore previous weights
#     cmd_train = "CUDA_VISIBLE_DEVICES={gpu} {python} train.py --model_dir={model_dir} --data_dir={data_dir}  --restore_file=best".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=data_dir)
#     print(cmd_train)
#     check_call(cmd_train, shell=True)

    # Launch evaluation with this config
    cmd_eval = "CUDA_VISIBLE_DEVICES={gpu} {python} evaluate.py --model_dir={model_dir} --data_dir={data_dir}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=eval_data_dir)
    print(cmd_eval)
    check_call(cmd_eval, shell=True)

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
    def multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[25],hidden_dims=[40],jet_numbers=[20000],Nfeatures=7,dir_name=None,name=None,info=None):
    
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
                  params.info=info

                  #-----------------------------------------
                  # Launch job (name has to be unique)
                  job_name = str(name)+'_lr_'+str(learning_rate)+'_decay_'+str(decay)+'_batch_'+str(batch_size)+'_epochs_'+str(num_epoch)+'_hidden_'+str(hidden_dim)+'_Njets_'+str(jet_number)+'_features_'+str(params.features)
                  
                  #Run the training - evaluation job
                  launch_training_job(parent_dir, args.data_dir, args.eval_data_dir, job_name, params, args.gpu)        

    #---------------------------------------------------------------------------------------------------------
    # SCANS OVER THE HYPERPARAMETER SPACE        
    
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[1],num_epochs=[1],hidden_dims=[40],jet_numbers=[10],dir_name='test_preprocess',name='jetpT_kt_') #gpu1
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[1],num_epochs=[1],hidden_dims=[50],jet_numbers=[10], Nfeatures=7,dir_name='join_tests/order_pT',name=sample_name.split('_')[-1],info='pT_order') #gpu0
    
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[1],num_epochs=[1],hidden_dims=[50],jet_numbers=[10], Nfeatures=7,dir_name='simple_rnn_kt_top_qcd',name='test_arch') #gpu0


#     multi_scan(learning_rates=[1e-3],decays=[0.9], batch_sizes=[10],num_epochs=[1],hidden_dims=[40],jet_numbers=[500],dir_name='test_simple_rnn_antikt',name='lr_batch_') #gpu0
#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,5e-4],decays=[0.9], batch_sizes=[1024],num_epochs=[1],hidden_dims=[40],jet_numbers=[40000],dir_name='test_simple_rnn_antikt',name='lr_batch_') #gpu0

#----------------------
# lr_batch_scan

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50],jet_numbers=[80000],dir_name='simple_rnn_kt_top_qcd',name='fully_connected_layers_100_200_50_1') #gpu0
# 
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[40],jet_numbers=[160000],dir_name='simple_rnn_kt_top_qcd',name='fully_connected_layers_50_50_25_1') #gpu0


    multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[50],jet_numbers=[80000], Nfeatures=7,dir_name='simple_rnn_top_qcd/desc_pT',name=sample_name.split('_')[-1],info='desc_pT') #gpu0

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[45],hidden_dims=[50],jet_numbers=[600000],dir_name='simple_rnn_kt_top_qcd',name='lr_batch_') #gpu2

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[40],hidden_dims=[40],jet_numbers=[160000],dir_name='simple_rnn_kt_top_qcd',name='lr_batch_') #gpu2
    
#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,1e-3],decays=[0.8], batch_sizes=[64,128,256,512,1024],num_epochs=[35],hidden_dims=[40],jet_numbers=[40000],dir_name='simple_rnn_kt_top_qcd',name='lr_batch_')#gpu1
#     
#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,5e-4],decays=[0.7], batch_sizes=[64,128,256,512,1024],num_epochs=[30],hidden_dims=[40],jet_numbers=[40000],dir_name='simple_rnn_kt_top_qcd',name='lr_batch_')#gpu2

#----------------------
#----------------------
# lr_batch_scan

#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,5e-4],decays=[0.9], batch_sizes=[512],num_epochs=[30],hidden_dims=[40],jet_numbers=[40000],dir_name='simple_rnn_kt',name='lr_batch_') #gpu0
    
#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,5e-4],decays=[0.8], batch_sizes=[512],num_epochs=[30],hidden_dims=[40],jet_numbers=[40000],dir_name='simple_rnn_kt',name='lr_batch_')#gpu1
#     
#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,5e-4],decays=[0.7], batch_sizes=[512],num_epochs=[30],hidden_dims=[40],jet_numbers=[40000],dir_name='simple_rnn_kt',name='lr_batch_')#gpu2

#----------------------
#----------------------
# h_dim_scan

#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[128],num_epochs=[30],hidden_dims=[40,80,160,320,640],jet_numbers=[40000],dir_name='simple_rnn_kt_top_qcd',name='h_dim_') #gpu0
    
###     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[128],num_epochs=[30],hidden_dims=[80,160,320,640],jet_numbers=[40000],dir_name='simple_rnn_kt_top_qcd',name='h_dim_')#gpu1
#     
#     multi_scan(learning_rates=[5e-3],decays=[0.9], batch_sizes=[1024],num_epochs=[30],hidden_dims=[40,80,160,320,640],jet_numbers=[40000],dir_name='simple_rnn_kt_top_qcd',name='h_dim_')#gpu1
    
#     multi_scan(learning_rates=[2e-3],decays=[0.7], batch_sizes=[512],num_epochs=[30],hidden_dims=[20],jet_numbers=[40000],dir_name='simple_rnn_kt_top_qcd',name='h_dim_')#gpu2
#-------------------------





#     multi_scan()
#     multi_scan(learning_rates=[1e-2, 5e-3,2e-3,5e-4],decays=[0.9,0.8,0.7], batch_sizes=[128],num_epochs=[25],hidden_dims=[40],jet_numbers=[40000],name='learning_rate_decay')

#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[64,128,256,512,1024],num_epochs=[25],hidden_dims=[40],jet_numbers=[40000],name='batch_size')   
    

#     multi_scan(learning_rates=[2e-3],decays=[0.9], batch_sizes=[20],num_epochs=[1],hidden_dims=[40],jet_numbers=[500],name='test_Njets')       
        
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
      
        