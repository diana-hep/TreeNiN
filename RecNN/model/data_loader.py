""" Classes and functions to load the raw data and create the batches """

import random
import numpy as np
import os
import sys
import pickle
import gzip
import subprocess
#import matplotlib as mpl
import json
import itertools
import re
import random
from sklearn.utils import check_random_state
import torch
from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler
import logging

from model import preprocess #The local dir is the train.py dir
##############################################################################################################
#/////////////////////   CLASSES     ////////////////////////////////////////////////////////////////////////
##############################################################################################################
# use GPU if available
class torch_params(object):

  cuda = torch.cuda.is_available()
  
##############################################################################################################
# methods to load the raw data and create the batches
class DataLoader(object):
  """
  Handles all aspects of the data. Has the methods to load the raw data and create the batches
  """
  def __init__(self):
    ''' Empty '''

  #-----------------------------------------------------------------------------------------------------------   
  # Make the input tree dictionaries. String should be either qcd or tt
  def makeTrees(dir_subjets,string,N_jets,label):
    '''
    Function to load the jet events and make the trees.
    Args:
      dir_subjets: dir with the event files. Loads all the files in the dir_subjets that satisfy the 'string' label. File format: array, where each entry is a "jet list". Each "jet list" has:
          tree=np.asarray(event[0])
          content=np.asarray(event[1])
          mass=np.asarray(event[2])
          pt=np.asarray(event[3])
          Currently only works for 1 jet per even. Modify for full event studies.
      string: string that identifies which files to load (signal or background files)
      N_jets: Number of jet trees to generate. If set to inf, it will load all the jets in the files
      label: label where 1 is for signal and 0 for background
      
    '''
    subjetlist = [filename for filename in np.sort(os.listdir(dir_subjets)) if ('tree' in filename and string in filename and filename.endswith('.dat'))]
    N_analysis=len(subjetlist)
    logging.info('Number of jet files for '+str(string)+'='+str(N_analysis))
    logging.info('Loading '+str(string)+' jet files... ')
    logging.info(str(string)+' files list = '+str(subjetlist))

    Ntotjets=0
    final_trees=[]  
    jets=[]
    ##----------------------------------------------- 
    # loop over the files and the events in each file
    
    for ifile in range(N_analysis):
      for s in open(dir_subjets+'/'+subjetlist[ifile]):
        if (Ntotjets>N_jets): return jets
        else:
        
          event=json.loads(s)
      #        print('Full event tree = ',event[0])
          Ntotjets+=1
  #         print('Ntotjets = ', Ntotjets)
          if Ntotjets%10000==0: logging.info('Ntotjets='+str(Ntotjets))
          tree=np.asarray(event[0])
          content=np.asarray(event[1])
          mass=np.asarray(event[2])
          pt=np.asarray(event[3])
    
          tree=np.array([np.asarray(e).reshape(-1,2) for e in tree])
          content=np.array([np.asarray(e).reshape(-1,4) for e in content])
    
    #       print('tree = ',tree[0])
    #       print('content = ',content[0])
    #       print('mass =',mass)
    #       print('pt = ',pt)

  #         # # SANITY CHECK: Below we check that the tree contains the right location of each children subjet 
  #         ii = 3;
  #   #       print('Content = ',content[0])
  #         print('Content ',ii,' = ',content[0][ii])
  #         print('Children location =',tree[0][ii])
  #         print('Content ',ii,' by adding the 2 children 4-vectors= ',content[0][tree[0][ii,0]] 
  #         + content[0][tree[0][ii,1]])
  #         print('-------------------'*10)
  
          ##-----------------------------------------------
          event=[]
          # loop over the jets in each event. Currently loads only the 1st jet
          for i in range(1): #This only works for single jet studies. Modify for full events
      
            jet = {}
   
            jet["root_id"] = 0
            jet["tree"] = tree[i] #Labels for the jet constituents in the tree 
#             jet["content"] = np.reshape(content[i],(-1,4,1)) #Where content[i][0] is the jet 4-momentum, and the other entries are the jets constituents 4 momentum. Use this format if using TensorFlow
            jet["content"] = np.reshape(content[i],(-1,4)) # Use this format if using Pytorch
            jet["mass"] = mass[i]
            jet["pt"] = pt[i]
            jet["energy"] = content[i][0, 3]
  
            px = content[i][0, 0] #The jet is the first entry of content. And then we have (px,py,pz,E)
            py = content[i][0, 1]
            pz = content[i][0, 2]
            p = (content[i][0, 0:3] ** 2).sum() ** 0.5
    #         jet["Calc energy"]=(p**2+mass[i]**2)**0.5
            eta = 0.5 * (np.log(p + pz) - np.log(p - pz)) #pseudorapidity eta
            phi = np.arctan2(py, px)
   
            jet["eta"] = eta
            jet["phi"] = phi
    #         print('jet contents =', jet.items())
#     
#             #-----------------------------
# #             Preprocess
# 
#             #Ensure that the left sub-jet has always a larger pt than the right
#             jet= preprocess.permute_by_pt(jet)
#             
#             # Change the input variables
#             jet= preprocess.extract(jet)
    
    
    
            if label==1:
              jet["label"]=1
            else:
              jet["label"]=0
              
            # Append each jet dictionary      
            jets.append(jet)  
  #           event.append(jet) #Uncomment for full event studies       
  #         jets.append(event)  #Uncomment for full event studies
  
    logging.info('Number of jets ='+ str(len(jets)))
    logging.info('---'*20)
#     print('Number of trees =', len(final_trees))
      
    return jets
  #-----------------------------------------------------------------------------------------------------------
  # Split the sample into train, cross-validation and test
  def merge_shuffle_sample(sig, bkg):
    '''
    Function to split the sample into train, cross-validation and test with equal number of sg and bg events. Then shuffle each set.
    Args:
      sig: signal sample
      bkg: background sample
      train_frac_rel: fraction of data for the train set
      val_frac_rel: fraction of data for the validation set
      test_frac_rel: fraction of data for the test set
    '''
    logging.info('---'*20)
    logging.info('Loading and shuffling the trees ...')
  
    rndstate = random.getstate()
    random.seed(0)
    size=np.minimum(len(sig),len(bkg))
#     print('sg length=',len(sig))
  
    sig_label=np.ones((size),dtype=int)
    bkg_label=np.zeros((size),dtype=int)

    ##-----------------------------------------------
    # Concatenate sg and bg data
    X=np.concatenate((sig[0:int(size)],bkg[0:int(size)]))
    Y=np.concatenate((sig_label[0:int(size)],bkg_label[0:int(size)]))

    ##-----------------------------------------------
    # Shuffle the sets
    indices = check_random_state(1).permutation(len(X))
    X = X[indices]
    Y = Y[indices]

    
    ##----------------------------------------------- 
    X=np.asarray(X)
    Y=np.asarray(Y)
    
#     Uncomment below if we change the NN output to be of dim=1
#     train_y=np.asarray(train_y).reshape((-1,1))
#     dev_y=np.asarray(dev_y).reshape((-1,1))
#     test_y=np.asarray(test_y).reshape((-1,1))
  
    logging.info('X shape='+str(X.shape))

  
    return X, Y

#-----------------------------------------------------------------------------------------------------------
  # 
  def scale_features(jets):
  
  
    transformer = RobustScaler().fit(np.vstack([jet["content"] for jet in jets]))  # remove outliers
    
    for jet in jets:
        jet["content"] = transformer.transform(jet["content"])  # center and scale the data

    return jets

#-----------------------------------------------------------------------------------------------------------
  # 
  def get_transformer(jets):
  
  
    transformer = RobustScaler().fit(np.vstack([jet["content"] for jet in jets]))  # remove outliers


    return transformer

#-----------------------------------------------------------------------------------------------------------
  # 
  def transform_features(transformer,jets):

    for jet in jets:
        jet["content"] = transformer.transform(jet["content"])  # center and scale the data

    return jets

  #-----------------------------------------------------------------------------------------------------------
  # Split the sample into train, cross-validation and test
  def split_shuffle_sample(sig, bkg, train_frac_rel, val_frac_rel, test_frac_rel):
    '''
    Function to split the sample into train, cross-validation and test with equal number of sg and bg events. Then shuffle each set.
    Args:
      sig: signal sample
      bkg: background sample
      train_frac_rel: fraction of data for the train set
      val_frac_rel: fraction of data for the validation set
      test_frac_rel: fraction of data for the test set
    '''
    print('---'*20)
    print('Loading and shuffling the trees ...')
  
    rndstate = random.getstate()
    random.seed(0)
    size=np.minimum(len(sig),len(bkg))
#     print('sg length=',len(sig))
  
    sig_label=np.ones((size),dtype=int)
    bkg_label=np.zeros((size),dtype=int)
    ##-----------------------------------------------
    print('Creating train, val and test datasets ...')
    
    # Split data into train, val and test
    train_frac=train_frac_rel
    val_frac=train_frac+val_frac_rel
    test_frac=val_frac+test_frac_rel

    N_train=int(train_frac*size)
    Nval=int(val_frac*size)
    Ntest=int(test_frac*size)
    ##-----------------------------------------------
    # Concatenate sg and bg data
    train_x=np.concatenate((sig[0:N_train],bkg[0:N_train]))
    train_y=np.concatenate((sig_label[0:N_train],bkg_label[0:N_train]))
  
    dev_x=np.concatenate((sig[N_train:Nval],bkg[N_train:Nval]))
    dev_y=np.concatenate((sig_label[N_train:Nval],bkg_label[N_train:Nval]))
  
    test_x=np.concatenate((sig[Nval:Ntest],bkg[Nval:Ntest]))
    test_y=np.concatenate((sig_label[Nval:Ntest],bkg_label[Nval:Ntest]))

    ##-----------------------------------------------
    # Shuffle the sets
    indices_train = check_random_state(1).permutation(len(train_x))
    train_x = train_x[indices_train]
    train_y = train_y[indices_train]
  
    indices_dev = check_random_state(2).permutation(len(dev_x))
    dev_x = dev_x[indices_dev]
    dev_y = dev_y[indices_dev]
  
    indices_test = check_random_state(3).permutation(len(test_x))
    test_x = test_x[indices_test]
    test_y = test_y[indices_test]
    
    ##----------------------------------------------- 
    train_x=np.asarray(train_x)
    dev_x=np.asarray(dev_x)
    test_x=np.asarray(test_x)
    train_y=np.asarray(train_y)
    dev_y=np.asarray(dev_y)
    test_y=np.asarray(test_y)
    
#     Uncomment below if we change the NN output to be of dim=1
#     train_y=np.asarray(train_y).reshape((-1,1))
#     dev_y=np.asarray(dev_y).reshape((-1,1))
#     test_y=np.asarray(test_y).reshape((-1,1))
  
    print('Train shape=',train_x.shape)
    print('Val shape=',dev_x.shape)
    print('Test shape=',test_x.shape)
#     print('train=',train_x[0]['content'])

  
    return train_x, train_y, dev_x, dev_y, test_x, test_y

  #-----------------------------------------------------------------------------------------------------------
  # CURRENTLY NOT USED. SKIP TO THE NEXT METHOD. 
  # Batchization of the recursion. 
  #This creates batches without zero padding. 

  def batch_level_no_pad(jets):
      # Batch the recursive activations across all nodes of a same level
      # !!! Assume that jets have at least one inner node.
      #     Leads to off-by-one errors otherwise :(

      # Reindex node IDs over all jets
      #
      # jet_children: array of shape [n_nodes, 2]
      #     jet_children[node_id, 0] is the node_id of the left child of node_id
      #     jet_children[node_id, 1] is the node_id of the right child of node_id
      #
      # jet_contents: array of shape [n_nodes, n_features]
      #     jet_contents[node_id] is the feature vector of node_id (4-vector in our case)
    
      jet_children =np.vstack([jet['tree'] for jet in jets])
  #     print('jet_children=',jet_children)
  #     jet_children = np.vstack(jet_children) #We concatenate all the jets tree  into 1 tree
  #     print('jet_children=',jet_children)
      jet_contents = np.vstack([jet["content"] for jet in jets]) #We concatenate all the jet['contents'] into 1 array
  #     print('jet_contents=',jet_contents)
      n_nodes=len(jet_children)

      #---------------------
      # Level-wise traversal
      level_children = np.zeros((n_nodes, 4), dtype=np.int32) #Array with 4 features per node
      level_children[:, [0, 2]] -= 1 #We set features 0 and 2 to -1. Features 0 and 2 will be the position of the left and right children of node_i, where node_i is given by "contents[node_i]" and left child is "content[level_children[node,0]]"

  # 
  #     # SANITY CHECK 1: Below we check that the jet_children contains the right location of each children subjet 
  #     ii = -28
  #     print('Content ',ii,' = ',jet_contents[ii])
  #     print('Children location =',jet_children[ii])
  #     if jet_children[ii][0]==-1: print('The node is a leaf')
  #     else: print('Content ',ii,' by adding the 2 children 4-vectors= ',jet_contents[jet_children[ii,0]] 
  #     + jet_contents[jet_children[ii,1]])

      inners = []   # Inner nodes at level i       ---- The nodes that are not leaves are in this category (SM)
      outers = []   # Outer nodes at level i       ---- The leaves are in this category (SM)
      offset = 0

      for jet in jets: # We fill the inners and outers array where each row corresponds to 1 level. We have each jet next to each other, so each jet root is a new column at depth 0, the first children add 2 columns at depth 1, .... Then we save in "level_children" the position of the left(right) child in the inners (or outers) array at depth i. So the 4-vector of node_i would be e.g. content[outers[level_children[i,0]]
    
          queue = [(jet["root_id"], -1, True, 0)] #(node, parent position, is_left, depth) 
        
        
          while len(queue) > 0:
              node, parent, is_left, depth = queue.pop(0) #We pop the first element (This is expensive because we have to change the position of all the other tuples in the queue)

              if len(inners) < depth + 1:
                  inners.append([]) #We append an empty list (1 per level) when the first node of a level shows up. 
              if len(outers) < depth + 1:
                  outers.append([])

              # Inner node
              if jet_children[node, 0] != -1:#If node is not a leaf (it has a left child)
                  inners[depth].append(node+offset) #We append the node to the inner list at row=depth because it has children
                  position = len(inners[depth]) - 1 #position on the inners list of the last node we added
                  is_leaf = False

                  queue.append((jet_children[node+offset, 0], node+offset, True, depth + 1)) #Format: (node at the next level, parent node,"left", depth)
                  queue.append((jet_children[node+offset, 1], node+offset, False, depth + 1))

              # Outer node
              else: #If the node is a leaf
                  outers[depth].append(node+offset)
  #                 print('outers=',outers)
                  position = len(outers[depth]) - 1 #position on the outers list of the last node we added
                  is_leaf = True

              # Register node at its parent. We save the position of the left and right children in the inners (or outers) array (at depth=depth_parent+1)
              if parent >= 0:
                  if is_left:
                      level_children[parent, 0] = position #position of the left child in the inners (or outers) array (at depth=depth_parent+1)
                      level_children[parent, 1] = is_leaf #if True then the left child is a leaf => look in the outers array, else in the inners one
                  else:
                      level_children[parent, 2] = position
                      level_children[parent, 3] = is_leaf

          offset += len(jet["tree"]) # We need this offset to get the right location in the jet_children array of each jet root node


  # 
  #     # SANITY CHECK 2: Below we check that the level_children contains the right location of each children subjet 
  #     ii = 1 #location of the parent in the inner list at level_parent
  #     level_parent=0
  #     print('Root of jet #',ii+1,' location =',inners[level_parent][ii]) #The root is at level 0
  #     print('Content jet #',ii+1,'=',jet_contents[inners[level_parent][ii]])
  #     print('Children location:\n left=',inners[level_parent+1][level_children[inners[level_parent][ii],0]],'   right=',inners[level_parent+1][level_children[inners[level_parent][ii],2]])
  #     if level_children[inners[level_parent][ii],1]==True: print('The node is a leaf')
  #     else: print('Content ',inners[0][ii],' by adding the 2 children 4-vectors= ',jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],0]]] 
  #     + jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],2]]])
  # 
  #     print('Is leaf at level ', level_parent,' = ', level_children[inners[level_parent][::],1])


      # Reorganize levels[i] so that inner nodes appear first, then outer nodes
      levels = []
      n_inners = []
      contents = []

      prev_inner = np.array([], dtype=int)
      print('----'*20)
      for inner, outer in zip(inners, outers):
          print('inner=',inner)
          print('outer=',outer)
        
          n_inners.append(len(inner)) # We append the number of inner nodes in each level
          inner = np.array(inner, dtype=int)
          outer = np.array(outer, dtype=int)
          levels.append(np.concatenate((inner, outer))) #Append the inners and outers of each level
        
 
        
          left = prev_inner[level_children[prev_inner, 1] == 1] # level_children[prev_inner, 1] returns a list with 1 for left children at level prev_inner+1 that are leaves and 0 otherwise. Then prev_inner[level_children[prev_inner, 1] == 1] picks the nodes at level prev_inner whose left children are leaves. So left are all nodes level prev_inner whose left child (at level prev_inner+1) is a leaf.
          level_children[left, 0] += len(inner) #We apply an offset to "left" because we concatenated inner and outer, with inners coming first. So now we get the right position of the children that are leaves in the levels array.
          right = prev_inner[level_children[prev_inner, 3] == 1]
          level_children[right, 2] += len(inner)


          contents.append(jet_contents[levels[-1]]) # We append the 4-vector given by the nodes in the last row that we added to levels. This way we create a list of contents where each row corresponds to 1 level.
  #         Then, the position of the left and right children in the levels list, will also be the position of them in the contents list, which is given by level_children Note that level_children keeps the old indices arrangement.

          prev_inner = inner #This will be the inner of the previous level in the next loop


  #         print('level_children[prev_inner, 1] =',level_children[prev_inner, 1] )
  #         print('left=',left)
  #         print('right=',right)
  #         print('prev_inner=',prev_inner)
  #         print('contents=',contents)
  #         print('length contents=',len(contents))
  #         print('length levels =',len(levels))

  # 
  # #     # SANITY CHECK 3:
  #     ii = 1 #location of the parent in the inner list at level_parent
  #     level_parent=3
  #     print('Final rearrangement of jets in batches')
  #     print('Root of jet #',ii+1,' location =','level',level_parent,' pos:',ii) #The root is at level 0
  #     print('Content jet #',ii+1,'=',contents[level_parent][ii])
  #     print('Children location in the contents list','level',level_parent+1,'\n left=',level_children[levels[level_parent][ii],0],'   right=',level_children[levels[level_parent][ii],2])
  #     if level_children[[ii],1]==True: print('The node is a leaf')
  #     else: print('Content ','level',level_parent,' pos:',ii,' by adding the 2 children 4-vectors= ',contents[level_parent+1][level_children[levels[level_parent][ii],0]] 
  #     + contents[level_parent+1][level_children[levels[level_parent][ii],2]])



      # levels: list of arrays
      #     levels[i][j] is a node id at a level i in one of the trees
      #     inner nodes are positioned within levels[i][:n_inners[i]], while
      #     leaves are positioned within levels[i][n_inners[i]:]
      #
      # level_children: array of shape [n_nodes, 4]
      #     level_children[node_id, 0] is the position j in the next level of
      #         the left child of node_id
      #     level_children[node_id, 2] is the position j in the next level of
      #         the right child of node_id
      #
      # n_inners: list of shape len(levels)
      #     n_inners[i] is the number of inner nodes at level i, accross all
      #     trees
      #
      # contents: array of shape [n_levels, n_nodes, n_features]
      #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
      #     or node layers[i][j]
    
      print('n_inners[0]=',n_inners[0])
      return (levels, level_children[:, [0, 2]], n_inners, contents)
      






  #----------------------------------------------------------------------------------------------------------- 
  # Batchization of the recursion (USING G LOUPPE'S CODE). String should be either qcd or tt. Adding zero padding
  def batch_nyu_pad(jets,features):
      # Batch the recursive activations across all nodes of a same level
      # !!! Assume that jets have at least one inner node.
      #     Leads to off-by-one errors otherwise :(

      # Reindex node IDs over all jets
      #
      # jet_children: array of shape [n_nodes, 2]
      #     jet_children[node_id, 0] is the node_id of the left child of node_id
      #     jet_children[node_id, 1] is the node_id of the right child of node_id
      #
      # jet_contents: array of shape [n_nodes, n_features]
      #     jet_contents[node_id] is the feature vector of node_id
      jet_children = []
      offset = 0

      for jet in jets:
          tree = np.copy(jet["tree"])
          tree[tree != -1] += offset #Everything except the leaves (SM)
          jet_children.append(tree)
          offset += len(tree) #I think this is the offset to go to the next jet and be able to train in parallel? (SM)

      jet_children = np.vstack(jet_children) #To get the tree of each jet one below the other (SM)
      jet_contents = np.vstack([jet["content"] for jet in jets])
      n_nodes = offset

      # Level-wise traversal
      level_children = np.zeros((n_nodes, 4), dtype=np.int32)
      level_children[:, [0, 2]] -= 1

      inners = []   # Inner nodes at level i       ---- The nodes that are not leaves are in this category (SM)
      outers = []   # Outer nodes at level i       ---- The leaves are in this category (SM)
      offset = 0

      for jet in jets:
          queue = [(jet["root_id"] + offset, -1, True, 0)]

          while len(queue) > 0:
              node, parent, is_left, depth = queue.pop(0)

              if len(inners) < depth + 1:
                  inners.append([])
              if len(outers) < depth + 1:
                  outers.append([])

              # Inner node
              if jet_children[node, 0] != -1:#If left child is not a leaf
                  inners[depth].append(node) #We append the node because it has children
                  position = len(inners[depth]) - 1
                  is_leaf = False

                  queue.append((jet_children[node, 0], node, True, depth + 1)) #Format: (left children position in contents, parent,"left", depth)
                  queue.append((jet_children[node, 1], node, False, depth + 1))

              # Outer node
              else:
                  outers[depth].append(node)
                  position = len(outers[depth]) - 1
                  is_leaf = True

              # Register node at its parent
              if parent >= 0:
                  if is_left:
                      level_children[parent, 0] = position #position of the left child in the inners (or outers) array
                      level_children[parent, 1] = is_leaf #if True look in the outers array, else in the inners one
                  else:
                      level_children[parent, 2] = position
                      level_children[parent, 3] = is_leaf

          offset += len(jet["tree"])

      # Reorganize levels[i] so that inner nodes appear first, then outer nodes
      levels = []
      n_inners = []
      contents = []
      n_level=[]
      prev_inner = np.array([], dtype=int)

      for inner, outer in zip(inners, outers):
          n_inners.append(len(inner))
          inner = np.array(inner, dtype=int)
          outer = np.array(outer, dtype=int)
          levels.append(np.concatenate((inner, outer)))
          n_level.append(len(levels[-1]))

          left = prev_inner[level_children[prev_inner, 1] == 1]
          level_children[left, 0] += len(inner)
          right = prev_inner[level_children[prev_inner, 3] == 1]
          level_children[right, 2] += len(inner)

          contents.append(jet_contents[levels[-1]])

          prev_inner = inner
     
#       print('----'*20)
#       print('subjets per level=',n_level)
#       print('----'*20)
#       print('----'*20)
#       print('Number of levels=',len(n_level))
#       print('----'*20)
      ##-----------------------------------------------
      # Zero padding
      #We loop over the levels to zero pad the array (only a few levels per jet)
      n_inners=np.asarray(n_inners)
      max_n_level=np.max(n_level)
  #     print('max_n_level=',max_n_level)
#       print('----'*20)

      for i in range(len(levels)): 
  #       print('max_n_level-len(levels[i])=',max_n_level-len(levels[i]))
        pad_dim=int(max_n_level-len(levels[i]))
        levels[i]=np.concatenate((levels[i],np.zeros((pad_dim))))                 
#         print('/////'*20)
#         print('contents[i].shape=',contents[i].shape)
        contents[i]=np.concatenate((contents[i],np.zeros((pad_dim,int(features)))))

      ##-----------------------------------------------


      # levels: list of arrays
      #     levels[i][j] is a node id at a level i in one of the trees
      #     inner nodes are positioned within levels[i][:n_inners[i]], while
      #     leaves are positioned within levels[i][n_inners[i]:]
      #
      # level_children: array of shape [n_nodes, 2]
      #     level_children[node_id, 0] is the position j in the next level of
      #         the left child of node_id
      #     level_children[node_id, 1] is the position j in the next level of
      #         the right child of node_id
      #
      # n_inners: list of shape len(levels)
      #     n_inners[i] is the number of inner nodes at level i, accross all
      #     trees
      #
      # contents: array of shape [n_nodes, n_features]
      #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
      #     or node layers[i][j]

#       return (levels, level_children[:, [0, 2]], n_inners, contents)
      return (levels, level_children[:, [0, 2]], n_inners, contents, n_level)



  #----------------------------------------------------------------------------------------------------------- 
  # CURRENTLY NOT USED. SKIP TO THE NEXT METHOD.
  # Batchization of the recursion with zero padding. 
  def batch_level(jets,features):
      '''
      This methos loads the jet trees, reorganizes the tree by levels, creates a batch of N jets by appending the nodes of each jet to each level and adds zero padding so that all the levels have the same size
      Args:
        jets: Number of jets to create the batch  
         
      ##-----------------------------------------------
        Batch the recursive activations across all nodes of a same level
        !!! Assume that jets have at least one inner node.
            Leads to off-by-one errors otherwise :(

        Reindex node IDs over all jets
      
        jet_children: array of shape [n_nodes, 2]
            jet_children[node_id, 0] is the node_id of the left child of node_id
            jet_children[node_id, 1] is the node_id of the right child of node_id
      
        jet_contents: array of shape [n_nodes, n_features]
            jet_contents[node_id] is the feature vector of node_id (4-vector in our case)
      '''
    
      jet_children =np.vstack([jet['tree'] for jet in jets])
  #     print('jet_children=',jet_children)
  #     jet_children = np.vstack(jet_children) #concatenate all the jets tree into 1 tree
  #     print('jet_children=',jet_children)
      jet_contents = np.vstack([jet["content"] for jet in jets]) #concatenate all the jet['contents'] into 1 array
  #     print('jet_contents=',jet_contents)
      n_nodes=len(jet_children)

      ##-----------------------------------------------
      # Level-wise traversal
      level_children = np.zeros((n_nodes, 4), dtype=np.int32) #Array with 4 features per node
      level_children[:, [0, 2]] -= 1 #We set features 0 and 2 to -1. Features 0 and 2 will be the position of the left and right children of node_i, where node_i is given by "contents[node_i]" and left child is "content[level_children[node,0]]"
      ##-----------------------------------------------

  #     # SANITY CHECK 1: Below we check that the jet_children contains the right location of each children subjet 
  #     ii = -28
  #     print('Content ',ii,' = ',jet_contents[ii])
  #     print('Children location =',jet_children[ii])
  #     if jet_children[ii][0]==-1: print('The node is a leaf')
  #     else: print('Content ',ii,' by adding the 2 children 4-vectors= ',jet_contents[jet_children[ii,0]] 
  #     + jet_contents[jet_children[ii,1]])
  
      ##-----------------------------------------------
      inners = []   # Inner nodes at level i       ---- The nodes that are not leaves are in this category 
      outers = []   # Outer nodes at level i       ---- The leaves are in this category 
      offset = 0
      
      # We fill the inners and outers array where each row corresponds to 1 level. We have each jet next to each other, so each jet root is a new column at depth 0, the first children add 2 columns at depth 1, and so on .... Then we save in "level_children" the position of the left(right) child in the inners or outers array at depth i. So the 4-vector of node_i would be e.g. content[outers[level_children[i,0]]
      for jet in jets: 
    
          queue = [(jet["root_id"], -1, True, 0)] #(node, parent position, is_left, depth) 
        
          while len(queue) > 0:
              node, parent, is_left, depth = queue.pop(0) #We pop the first element (This is expensive because we have to change the position of all the other tuples in the queue)

              if len(inners) < depth + 1:
                  inners.append([]) #We append an empty list (1 per level) when the first node of a level shows up. 
              if len(outers) < depth + 1:
                  outers.append([])
              #-----------
              # Inner node
              if jet_children[node, 0] != -1:#If node is not a leaf (it has a left child)
                  inners[depth].append(node+offset) #We append the node to the inner list at row=depth because it has children
                  position = len(inners[depth]) - 1 #position on the inners list of the last node we added
                  is_leaf = False

                  queue.append((jet_children[node+offset, 0], node+offset, True, depth + 1)) #Format: (node at the next level, parent node,"left", depth)
                  queue.append((jet_children[node+offset, 1], node+offset, False, depth + 1))
              #-----------
              # Outer node
              else: #If the node is a leaf
                  outers[depth].append(node+offset)
  #                 print('outers=',outers)
                  position = len(outers[depth]) - 1 #position on the outers list of the last node we added
                  is_leaf = True
                  
              #-----------
              # Register node at its parent. We save the position of the left and right children in the inners (or outers) array (at depth=depth_parent+1)
              if parent >= 0:
                  if is_left:
                      level_children[parent, 0] = position #position of the left child in the inners (or outers) array (at depth=depth_parent+1)
                      level_children[parent, 1] = is_leaf #if True then the left child is a leaf => look in the outers array, else in the inners one
                  else:
                      level_children[parent, 2] = position
                      level_children[parent, 3] = is_leaf

          offset += len(jet["tree"]) # We need this offset to get the right location in the jet_children array of each jet root node because we concatenate one jet after each other
      ##-----------------------------------------------

  #     # SANITY CHECK 2: Below we check that the level_children contains the right location of each children subjet 
  #     ii = 1 #location of the parent in the inner list at level_parent
  #     level_parent=0
  #     print('Root of jet #',ii+1,' location =',inners[level_parent][ii]) #The root is at level 0
  #     print('Content jet #',ii+1,'=',jet_contents[inners[level_parent][ii]])
  #     print('Children location:\n left=',inners[level_parent+1][level_children[inners[level_parent][ii],0]],'   right=',inners[level_parent+1][level_children[inners[level_parent][ii],2]])
  #     if level_children[inners[level_parent][ii],1]==True: print('The node is a leaf')
  #     else: print('Content ',inners[0][ii],' by adding the 2 children 4-vectors= ',jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],0]]] 
  #     + jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],2]]])
  # 
  #     print('Is leaf at level ', level_parent,' = ', level_children[inners[level_parent][::],1])

      ##-----------------------------------------------
      # Reorganize levels[i] so that inner nodes appear first, then outer nodes
      levels = []
      n_inners = []
      contents = []
      n_level=[]

      prev_inner = np.array([], dtype=int)

      for inner, outer in zip(inners, outers):       
#           print('inner=',inner)
#           print('outer=',outer)
          n_inners.append(len(inner)) # We append the number of inner nodes in each level
          inner = np.array(inner, dtype=int)
          outer = np.array(outer, dtype=int)
          levels.append(np.concatenate((inner, outer))) #Append the inners and outers of each level
          n_level.append(len(levels[-1]))
  
          left = prev_inner[level_children[prev_inner, 1] == 1] # level_children[prev_inner, 1] returns a list with 1 for left children at level prev_inner+1 that are leaves and 0 otherwise. Then prev_inner[level_children[prev_inner, 1] == 1] picks the nodes at level prev_inner whose left children are leaves. So left are all nodes level prev_inner whose left child (at level prev_inner+1) is a leaf.
          level_children[left, 0] += len(inner) #We apply an offset to "left" because we concatenated inner and outer, with inners coming first. So now we get the right position of the children that are leaves in the levels array.
          right = prev_inner[level_children[prev_inner, 3] == 1]
          level_children[right, 2] += len(inner)


          contents.append(jet_contents[levels[-1]]) # We append the 4-vector given by the nodes in the last row that we added to levels. This way we create a list of contents where each row corresponds to 1 level.
  #         Then, the position of the left and right children in the levels list, will also be the position of them in the contents list, which is given by level_children Note that level_children keeps the old indices arrangement.

          prev_inner = inner #This will be the inner of the previous level in the next loop

  #         print('level_children[prev_inner, 1] =',level_children[prev_inner, 1] )
  #         print('left=',left)
  #         print('right=',right)
  #         print('prev_inner=',prev_inner)
  #         print('contents=',contents)
  #         print('length contents=',len(contents))
  #         print('length levels =',len(levels))

      ##-----------------------------------------------

  # #     # SANITY CHECK 3:
  #     ii = 1 #location of the parent in the inner list at level_parent
  #     level_parent=3
  #     print('Final rearrangement of jets in batches')
  #     print('Root of jet #',ii+1,' location =','level',level_parent,' pos:',ii) #The root is at level 0
  #     print('Content jet #',ii+1,'=',contents[level_parent][ii])
  #     print('Children location in the contents list','level',level_parent+1,'\n left=',level_children[levels[level_parent][ii],0],'   right=',level_children[levels[level_parent][ii],2])
  #     if level_children[[ii],1]==True: print('The node is a leaf')
  #     else: print('Content ','level',level_parent,' pos:',ii,' by adding the 2 children 4-vectors= ',contents[level_parent+1][level_children[levels[level_parent][ii],0]] 
  #     + contents[level_parent+1][level_children[levels[level_parent][ii],2]])
    
    
#     
#       ##-----------------------------------------------
#       # Zero padding
#       #We loop over the levels to zero pad the array (only a few levels per jet)
#       n_inners=np.asarray(n_inners)
#       max_n_level=np.max(n_level)
#   #     print('max_n_level=',max_n_level)
# #       print('----'*20)
# 
#       for i in range(len(levels)): 
#   #       print('max_n_level-len(levels[i])=',max_n_level-len(levels[i]))
#         pad_dim=int(max_n_level-len(levels[i]))
#         levels[i]=np.concatenate((levels[i],np.zeros((pad_dim))))  
#         contents[i]=np.concatenate((contents[i],np.zeros((pad_dim,int(features)))))
# 
#       ##-----------------------------------------------
      '''
      levels: list of arrays
          levels[i][j] is a node id at a level i in one of the trees
          inner nodes are positioned within levels[i][:n_inners[i]], while
          leaves are positioned within levels[i][n_inners[i]:]
      
      level_children: array of shape [n_nodes, 4]
          level_children[node_id, 0] is the position j in the next level of
              the left child of node_id
          level_children[node_id, 2] is the position j in the next level of
              the right child of node_id
      
      n_inners: list of shape len(levels)
          n_inners[i] is the number of inner nodes at level i, accross all
          trees
      
      contents: array of shape [n_levels, n_nodes, n_features]
          contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
          or node layers[i][j]
          
      n_level: list with the number of nodes in each level
      '''
      # return (levels, level_children[:, [0, 2]], n_inners, contents, n_level)
      return (levels, level_children[:, [0, 2]], n_inners, contents)
  
  #----------------------------------------------------------------------------------------------------------- 
  # Generator function
  def make_pad_batch_iterator_level( batches, batch_size):
    '''
     This method is a generator function that loads the batches, shifts numpy arrays to torch tensors and feeds the training, val pipeline
     Args:
      batches: batches of data
      batch_size: number of jets per batch
    '''
     
    for i in range(len(batches)):
    
      levels = np.asarray(batches[i][0])
      children = np.asarray(batches[i][1]) # Children is an array with shape = (total n_nodes in the batch,2) where shape[1] contains the left and right children locations of the node. This location gives the position of the children in the next level
      n_inners = np.asarray(batches[i][2])
      contents = np.asarray(batches[i][3])
      n_level = np.asarray(batches[i][4])
      labels= np.asarray(batches[i][5])
    
#       print('levels=',levels)
#       print('----'*20)
#       print('children=',children)
#       print('----'*20)
#       print('n_inners=',n_inners)
#       print('----'*20)
#       print('contents=',contents)
#       print('----'*20)
#       print('n_level=',n_level)
#       print('----'*20)
#       print('labels=',labels)
    
      levels=torch.LongTensor(levels)
      children=torch.LongTensor(children)
      n_inners=torch.LongTensor(n_inners)
      contents = torch.FloatTensor(contents)
      n_level=torch.LongTensor(n_level)
      labels= torch.LongTensor(labels)
      
      ##-----------------------------------------------
      # shift tensors to GPU if available
      if torch_params.cuda:
        levels = levels.cuda()
        children=children.cuda()
        n_inners=n_inners.cuda()
        contents=contents.cuda()
        n_level= n_level.cuda()
        labels =labels.cuda()
        
      ##-----------------------------------------------
      # convert them to Variables to record operations in the computational graph

      levels=Variable(levels)
      children=Variable(children)
      n_inners=Variable(n_inners)
      contents = Variable(contents)
      n_level=Variable(n_level)
      labels = Variable(labels)
      
      
      yield levels, children, n_inners, contents, n_level, labels


##############################################################################################################
#/////////////////////      OTHER FUNCTIONS     //////////////////////////////////////////////////////////////
##############################################################################################################
# Loads the DataLoader class to create the train, val, test datasets with zero paddings
def batch_array(sample_x,sample_y,batch_size, features):
  '''
  Loads the DataLoader class to create the train, val, test datasets
  Args:
    sample_x: jet trees 
    sample_y: truth value for the jet labels
    batch_size: number of jets in each batch
  '''
  tot_levels=[]
  
  loader=DataLoader
  num_steps=len(sample_x)//batch_size
  batches=[]
  for i in range(num_steps):
    batches.append([])
    levels, children, n_inners, contents, n_level= loader.batch_nyu_pad(sample_x[i*batch_size:(i+1)*batch_size],features)
    batches[-1].append(levels)
    batches[-1].append(children)
    batches[-1].append(n_inners)
    batches[-1].append(contents)
    batches[-1].append(n_level)
    batches[-1].append(sample_y[i*batch_size:(i+1)*batch_size])
    if (i+1)%100==0: logging.info('Number of batches created='+str(i+1))

#     
#     #Get average number of levels
#     tot_levels.append(n_level)
#     
#   print('Total jets=',len(tot_levels))
#   print('----'*20)
#   print('Average levels per jet=',np.sum([len(level) for level in tot_levels])/len(tot_levels))
    
  batches=np.asarray(batches)
  
  return batches


#-------------------------------------------------------------------------------------------------------------
# CURRENTLY NOT USED. SKIP TO THE NEXT FUNCTION
# Loads the DataLoader class to create the train, val, test datasets without zero padding
def batch_array_no_pad(sample_x,sample_y,batch_size):
  '''
  Loads the DataLoader class to create the train, val, test datasets without zero padding
  Args:
    sample_x: jet trees 
    sample_y: truth value for the jet labels
    batch_size: number of jets in each batch
  '''
  loader=DataLoader
  num_steps=len(sample_x)//batch_size
  batches=[]
  for i in range(num_steps):
    batches.append([])
    levels, children, n_inners, contents= loader.batch_level_no_pad(sample_x[i*batch_size:(i+1)*batch_size])
    batches[-1].append(levels)
    batches[-1].append(children)
    batches[-1].append(n_inners)
    batches[-1].append(contents)
    batches[-1].append(sample_y[i*batch_size:(i+1)*batch_size])
    
  batches=np.asarray(batches)
  
  return batches
  
  
#-------------------------------------------------------------------------------------------------------------
###///////////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------            
# if __name__=='__main__':
#   
#   myN_jets=10
#   batch_size=1
#   
#   load=DataLoader
# 
#   sig_tree, sig_list=load.makeTrees(dir_jets_subjets,sg,myN_jets,0)
#   bkg_tree, bkg_list=load.makeTrees(dir_jets_subjets,bg,myN_jets,0)  
#   
#   train_data, dev_data, test_data = load.shuffle_split(sig_list, bkg_list, 0.6, 0.2, 0.2) 
#   
#   data_iterator=load.make_pad_batch_iterator(train_data, batch_size)
  

  
  
  
  