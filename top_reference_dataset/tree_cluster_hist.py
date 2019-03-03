#!/usr/bin/env python

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import copy

#-------------------------------------------------------------------------------------------------------------
#/////////////////////     FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------    
# Recursive function to go deep up to the leaves and then go back to the root adding the children values to get the parent ones
def rewrite_feature(feature,tree):
    feature = copy.deepcopy(feature)
    tree = copy.deepcopy(tree)

    def _rec(i):
#         print('tree[i, 0]=',tree[2*i])
        if tree[2*i] == -1:
            pass
        else:
            _rec(tree[2*i]) #We call the function recursively before adding the children values => We will start from the leaves and go up to the root. When we hit a leaf, the 'pass' statement will just pass and so we will get 'c'
            _rec(tree[2*i+ 1])
            c = feature[tree[2*i]] + feature[tree[2*i+ 1]] #We add the 4-vectors of the two children of object i 
#             print('/////'*20)
#             print('feature[i] before=',feature[i])
            feature[i] = c # We replace the 4-vector of object i by the sum of the 4-vector of its children
#             print('feature[i] after=',feature[i])
            
    _rec(0) # We start form the root id

#     if jet["content"].shape[1] == 5:
#         jet["content"][:, 4] = pflow
#     print('/////'*20)
#     print('/////'*20)
#     print('jet=',jet)
    return feature


#------------------------------------------------------------------------------------------------------------- 
# Recursive function to access fastjet clustering history and make the tree. We will call this function below in _traverse.
def _traverse_rec(root, parent_id, is_left, tree, content, charge, abs_charge, muon, extra_info=True): #root should be a fj.PseudoJet
  
  id=len(tree)/2
  if parent_id>=0:
    if is_left:
       tree[2 * parent_id] = id #We set the location of the lef child in the content array of the 4-vector stored in content[parent_id]. So the left child will be content[tree[2 * parent_id]]
    else:
       tree[2 * parent_id + 1] = id #We set the location of the right child in the content array of the 4-vector stored in content[parent_id]. So the right child will be content[tree[2 * parent_id+1]]
#  This is correct because with each 4-vector we increase the content array by one element and the tree array by 2 elements. But then we take id=tree.size()//2, so the id increases by 1. The left and right children are added one after the other.

  #-------------------------------
  # We insert 2 new nodes to the vector that constitutes the tree. In the next iteration we will replace this 2 values with the location of the parent of the new nodes
  tree.append(-1)
  tree.append(-1)
    
#     We fill the content vector with the values of the node 
  content.append(root.px())
  content.append(root.py())
  content.append(root.pz())
  content.append(root.e())

  #--------------------------------------
  # We move from the root down until we get to the leaves. We do this recursively
  if root.has_pieces():
    if extra_info:
      charge.append('inner')
      abs_charge.append('inner')
      muon.append('inner')
    
#     print('---'*20)
#     print('tree=',tree)
#     print('length tree=',len(tree))
#     print('---'*20)
#   #               print('content=',content)
#     print('---'*20)
#     print('charge=',charge)
#     print('length charge=',len(charge))
#     print('---'*20)

    #------------------------------   
    # Call the function recursively 
    pieces = root.pieces()
    _traverse_rec(pieces[0], id, True, tree, content,charge,abs_charge,muon, extra_info=extra_info) #pieces[0] is the left child
    _traverse_rec(pieces[1], id, False, tree, content,charge,abs_charge,muon, extra_info=extra_info) #pieces[1] is the right child
      
  else:
    if extra_info:
      charge.append(root.python_info().Charge)
      abs_charge.append(np.absolute(root.python_info().Charge))
      muon.append(root.python_info().Muon)
    
#     print('abs_charge=',np.absolute(root.python_info().Charge))
    
#     print('---'*20)
#     print('tree=',tree)
#     print('length tree=',len(tree))
#     print('---'*20)
#   #               print('content=',content)
#     print('---'*20)
#     print('charge=',charge)
#     print('length charge=',len(charge))
#     print('---'*20) 
  
#   print('---'*20)
#   print('tree=',tree)
#   print('length tree=',len(tree))
#   print('---'*20)
# #               print('content=',content)
#   print('---'*20)
#   print('charge=',charge)
#   print('length charge=',len(charge))
#   print('---'*20) 
#   print('length content=',len(content)/4)
  
  
#------------------------------------------------------------------------------------------------------------- 
# This function call the recursive function to make the trees starting from the root
def _traverse(root, extra_info=True):#root should be a fj.PseudoJet
  tree=[]
  content=[]
  charge=[]
  abs_charge=[]
  muon=[]
#   sum_abs_charge=0
  _traverse_rec(root, -1, False, tree, content,charge,abs_charge, muon, extra_info=extra_info) #We start from the root=jet 4-vector
  return tree, content, charge, abs_charge, muon

#------------------------------------------------------------------------------------------------------------- 
