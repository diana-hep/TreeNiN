import os, sys


Nstart=sys.argv[1] # Jet number starting point. Set to 0 to start from the beginning. Set to 400000 to get the last 4000 jets only (1% of the dataset)


#Read dataset and get non-zero entries with 4-vector info
os.chdir('/TreeNiN/code/top_reference_dataset/')

if len(sys.argv) > 2:
  test_set=sys.argv[2]
  os.system('python3 ReadData.py '+str(Nstart)+' '+test_set)
else:
  os.system('python3 ReadData.py '+str(Nstart))


# Create the input trees. 
# Load and recluster the jet constituents. Create binary trees with the clustering history of the jets and output a dictionary for each jet that contains the root_id, tree, content (constituents 4-momentum vectors), mass, pT, energy, eta and phi values (also charge, muon ID, etc depending on the information contained in the dataset)
# FastJet needs python2.7
os.system('python2.7 toptag_reference_dataset_Tree.py jet_image_trim_pt800-900_card.dat test out_data/ ../data/inputTrees/top_tag_reference_dataset/')

# Apply preprocessing: get the initial 7 features: p, eta, phi, E, E/JetE, pT, theta. Apply RobustScaler
os.chdir('/TreeNiN/code/recnn/')
os.system('python3 run_preprocess.py')
