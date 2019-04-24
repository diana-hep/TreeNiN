#!/usr/bin/env python
## USAGE --- python ReadData.py N_start
# from __future__ import print_function

import pandas
import sys, os
import numpy as np
import pickle
import urllib.request

# Starting jet to read from the dataset
N_start=int(sys.argv[1])

# print(len(sys.argv))


### This function reads the h5 files and saves the jets in numpy arrays. We only save the non-zero 4-vector entries
def h5_to_npy(filename,Njets,Nstart=None):
    file = pandas.HDFStore(filename)
    jets=np.array(file.select("table",start=Nstart,stop=Njets))
    jets2=jets[:,0:800].reshape((len(jets),200,4)) #This way I'm getting the 1st 199 constituents. jets[:,800:804] is the constituent 200. jets[:,804] has a label=0 for train, 1 for test, 2 for val. jets[:,805] has the label sg/bg
#     print('jets2=',jets2[0])
    labels=jets[:,805:806]
#     print('labels=',labels)
    npy_jets=[]
    for i in range(len(jets2)):
#         print('~np.all(jets2[i] == 0, axis=1)=',~np.all(jets2[i] == 0, axis=1))
        # Get the index of non-zero entries
        nonzero_entries=jets2[i][~np.all(jets2[i] == 0, axis=1)]
        npy_jets.append([nonzero_entries,0 if labels[i] == 0 else 1])
        
    file.close()
    return npy_jets
    
#--------------------------------------

if len(sys.argv) > 2:
  test_set=str(sys.argv[2])
else:
  print('Beginning file download ...')
  url = 'https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=test.h5'  
  urllib.request.urlretrieve(url, 'in_data/test.h5')

  test_set='in_data/test.h5'




print('Loading jet constituents ...')
print('==='*20)
  
test_data=h5_to_npy(test_set,None,Nstart=N_start)
with open('out_data/test_jets.pkl', "wb") as f: pickle.dump(test_data, f, protocol=2)












