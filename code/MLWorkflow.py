import os, sys

# Select start and finish model number for the scan. To evaluate the 9 models select Nstart=0 and Nfinish=9 
Nstart=sys.argv[1]
Nfinish=sys.argv[2]


os.chdir('/TreeNiN/code/recnn/')

# Load the trained weights and evaluate each model 
os.system('python3 search_hyperparams.py --NrunStart='+str(Nstart)+' --NrunFinish='+str(Nfinish))
