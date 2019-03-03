import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.autograd import Variable






##############################################################################################################
#/////////////////////   CLASSES     ////////////////////////////////////////////////////////////////////////
##############################################################################################################

#-------------------------------------------------------------------------------------------------------------
# 0) Leaves/inners different weights RecNN
#-------------------------------------------------------------------------------------------------------------
# Make embedding
class GRNNTransformLeaves(nn.Module):
  '''
  Recursive neural network architecture. nn.Module is the NN superclass in torch. First creates the recursive graph and then concatenates a fully-connected NN that ends with 2 nodes (binary classification).
  '''

  def __init__(self, params, features=None, hidden=None,**kwargs):
    super(GRNNTransformLeaves,self).__init__()
        
    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc_u = nn.Linear(params.features, params.hidden)
    self.fc_h = nn.Linear(3 * params.hidden, params.hidden) 

    self.fc_u_inner1 = nn.Linear(params.features, params.hidden)
    self.fc_u_outer1 = nn.Linear(params.features, params.hidden)
    
    self.fc_u_inner = nn.Linear(params.features, params.hidden)
    self.fc_u_outer = nn.Linear(params.features, params.hidden)
    
#     self.fc_h_inner = nn.Linear(3 * params.hidden, params.hidden) 
#     self.fc_h_outer = nn.Linear(3 * params.hidden, params.hidden)

    self.fc_N0 = nn.Linear(params.hidden, params.hidden)
    self.fc_N1 = nn.Linear(params.hidden, params.hidden)
    self.fc_N2 = nn.Linear(params.hidden, params.hidden)
    self.fc_N3 = nn.Linear(params.hidden, params.hidden)
    self.fc_N4 = nn.Linear(params.hidden, params.hidden)
    self.fc_N5 = nn.Linear(params.hidden, params.hidden)
    self.fc_N6 = nn.Linear(params.hidden, params.hidden)
    self.fc_N7 = nn.Linear(params.hidden, params.hidden)
    self.fc_N8 = nn.Linear(params.hidden, params.hidden)
    self.fc_N9 = nn.Linear(params.hidden, params.hidden)

    
#     fully_connected_neurons1=50 
#     fully_connected_neurons2=50
#     fully_connected_neurons3=25
#     self.fc1 = nn.Linear(params.hidden, fully_connected_neurons1)
#     self.fc2 = nn.Linear(fully_connected_neurons1, fully_connected_neurons2)
#     self.fc4 = nn.Linear(fully_connected_neurons2, fully_connected_neurons3)    
#     self.fc3 = nn.Linear(fully_connected_neurons3, params.number_of_labels_types)
        
       #Look at this as it didn't let the NN learn on some tests 
    gain = nn.init.calculate_gain(activation_string) # gain=sqrt[2] for ReLU
#     print('gain=',gain)
#     print('---'*20)
#     print('---'*20)
  


# 
# #   ##--------------------------------------------------
#     nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
#     nn.init.orthogonal_(self.fc_h.weight, gain=gain) 
#        
#     nn.init.xavier_uniform_(self.fc_N0.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N1.weight, gain=gain) 
#     nn.init.xavier_uniform_(self.fc_N2.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N3.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N4.weight, gain=gain)     
#     nn.init.xavier_uniform_(self.fc_N5.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N6.weight, gain=gain)


#   ##-----------------------------------------------------
    nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
    nn.init.orthogonal_(self.fc_h.weight, gain=gain) 


    nn.init.xavier_uniform_(self.fc_u_inner1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_u_outer1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_u_inner.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_u_outer.weight, gain=gain)
#     nn.init.orthogonal_(self.fc_h_inner.weight, gain=gain) 
#     nn.init.orthogonal_(self.fc_h_outer.weight, gain=gain) 


       
    nn.init.orthogonal_(self.fc_N0.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N1.weight, gain=gain) 
    nn.init.orthogonal_(self.fc_N2.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N3.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N4.weight, gain=gain)     
    nn.init.orthogonal_(self.fc_N5.weight, gain=gain)

    nn.init.orthogonal_(self.fc_N6.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N7.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N8.weight, gain=gain) 
    nn.init.orthogonal_(self.fc_N9.weight, gain=gain) 

#     fully_connected_neurons1=50 
#     fully_connected_neurons2=50
#     fully_connected_neurons3=25
#     self.fc1 = nn.Linear(params.hidden, fully_connected_neurons1)
#     self.fc2 = nn.Linear(fully_connected_neurons1, fully_connected_neurons2)
#     self.fc4 = nn.Linear(fully_connected_neurons2, fully_connected_neurons3)    
#     self.fc3 = nn.Linear(fully_connected_neurons3, params.number_of_labels_types)
        
       #Look at this as it didn't let the NN learn on some tests 
#     gain = nn.init.calculate_gain(activation_string)
#     nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
#     nn.init.orthogonal_(self.fc_h.weight, gain=gain) 
                
        
#     activation_string = 'relu'
#     self.activation = getattr(F, activation_string)

  ##-----------------------------------------------
  def forward(self, params, levels, children, n_inners, contents, n_level):

    n_levels = len(levels)
    embeddings = []
   
    #invert the levels using pytorch
    inv_idx = torch.arange(levels.size(0)-1, -1, -1).long()
    inv_levels = levels[inv_idx]     
      
    # loop over each level, starting from the bottom
    for i, nodes in enumerate(inv_levels):
      j = n_levels - 1 - i
      try:
          inner = nodes[:n_inners[j]]
      except ValueError:
          inner = []
      try:
          outer = nodes[n_inners[j]:n_level[j]]
      except ValueError:
          outer = []
#       u_k = self.fc_u(contents[j])
      
      
      
      
      
      
      #-------------------
      # SPLIT outer vs inner NODE
#       print('j=',j)
#       print('contents[j]=',contents[j])

#       u_k_inner=torch.FloatTensor([])
#       u_k_outer=torch.FloatTensor([])
      
#       if torch.cuda.is_available():
#         u_k_inner=u_k_inner.cuda()
#         u_k_outer=u_k_outer.cuda()
        
      if len(inner) > 0:
        u_k_inner = self.fc_u_inner(contents[j][:n_inners[j]])
#         print('n_inners[j]=',n_inners[j])
#         print('contents[:n_inners[j]]=',contents[j][:n_inners[j]])
#         print('inner=',inner)
        
      if len(outer)>0:
#         print('outer=',outer)
        u_k_outer = self.fc_u_outer(contents[j][n_inners[j]:n_level[j]])
#         print('u_k_outer shape=',u_k_outer.shape)
#       print('---'*20)
      
      if len(inner) > 0 and len(outer)>0:
        u_k=torch.cat((u_k_inner, u_k_outer), 0)
      elif len(inner) > 0:
        u_k=u_k_inner
      else:
        u_k=u_k_outer
        

#  
#       #---------------------
#       # Same inner outer weights          
#       u_k = self.fc_u(contents[j])
      
      
#       print('u_k=',u_k.shape)
#       print('===='*20) 
      
#       if j==6:
#         sys.exit()

      '''
          #Change the activation from RELU to 1X1 Convolution#
                '''
      u_k = self.activation(u_k)
  
      # implement the recursion  
      if len(inner) > 0:
          zero = torch.zeros(1).long(); one = torch.ones(1).long()
          if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
          h_L = embeddings[-1][children[inner, zero]]
          h_R = embeddings[-1][children[inner, one]]
        
          h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
          h = self.fc_h(h)
          '''
              #Change the activation from RELU to 1X1 Convolution#
              '''
          h = self.activation(h)
                
          try:
              embeddings.append(torch.cat((h, u_k[n_inners[j]:n_level[j]]), 0))
          except ValueError:
              embeddings.append(h)
      
      else:
        embeddings.append(u_k)

#     print('Length Embeddings=',len(embeddings))
#     print('embeddings[-1]=',embeddings[-1])
    
    h_out=embeddings[-1].view((params.batch_size, -1))
        
    return h_out

# --------------------------------------
# Classifier

class PredictFromParticleEmbeddingLeaves(GRNNTransformLeaves): #We call GRNNTransformSimple as the superclass

  def __init__(self, params, make_embedding=None, features=None, hidden=None,**kwargs):
    super().__init__(params, features, hidden,**kwargs) #particle_transform is the RecNN architecture, e.g. GRNNTransformSimple. We pass the arguments of the __init__ of the GRNNTransformSimple as input
  
    self.transform = make_embedding(params, features=features, hidden=hidden,**kwargs) 

    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc1 = nn.Linear(params.hidden, params.hidden)
    self.fc2 = nn.Linear(params.hidden, params.hidden)
    self.fc3 = nn.Linear(params.hidden, params.number_of_labels_types)

    gain = nn.init.calculate_gain(activation_string)
    nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc3.weight, gain=gain)
    nn.init.constant_(self.fc3.bias, 1)        
        
#     activation_string = 'relu'
#     self.activation = getattr(F, activation_string)

  ##-------------------------------------------------
#   def forward(self, params, levels, children, n_inners, contents, n_level):

  def forward(self, params, levels, children, n_inners, contents, n_level, **kwargs):
    h_out = self.transform(params, levels, children, n_inners, contents, n_level, **kwargs)

    ##-------------------
    #concatenate a fully-connected NN
    h_out = self.fc1(h_out)

    h_out = self.activation(h_out)

    h_out = self.fc2(h_out)

    h_out = self.activation(h_out)


  #     h_out = self.fc4(h_out)  
  #     h_out = self.activation(h_out)

  #     h_out = F.log_softmax(self.fc3(h_out), dim=1) # dim: batch_size*seq_len x num_tags

    output = torch.sigmoid(self.fc3(h_out))
    # output = F.sigmoid(self.fc3(h_out))


    return output 

#--------------------------------------------------------------------------------------------------------------
# 1) Network in Network -  RecNN
#-------------------------------------------------------------------------------------------------------------
class GRNNTransformSimpleNiN(nn.Module):
  '''
  Recursive neural network architecture. nn.Module is the NN superclass in torch. First creates the recursive graph and then concatenates a fully-connected NN that ends with 2 nodes (binary classification).
  '''

  def __init__(self, params, features=None, hidden=None,**kwargs):
    super(GRNNTransformSimpleNiN,self).__init__()
        
    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc_u = nn.Linear(params.features, params.hidden)
    self.fc_h = nn.Linear(3 * params.hidden, params.hidden)


    self.fc_u_inner1 = nn.Linear(params.features, params.hidden)
    self.fc_u_outer1 = nn.Linear(params.features, params.hidden)
    
    self.fc_u_inner = nn.Linear(params.features, params.hidden)
    self.fc_u_outer = nn.Linear(params.features, params.hidden)
    
#     self.fc_h_inner = nn.Linear(3 * params.hidden, params.hidden) 
#     self.fc_h_outer = nn.Linear(3 * params.hidden, params.hidden)

    self.fc_N0 = nn.Linear(params.hidden, params.hidden)
    self.fc_N1 = nn.Linear(params.hidden, params.hidden)
    self.fc_N2 = nn.Linear(params.hidden, params.hidden)
    self.fc_N3 = nn.Linear(params.hidden, params.hidden)
    self.fc_N4 = nn.Linear(params.hidden, params.hidden)
    self.fc_N5 = nn.Linear(params.hidden, params.hidden)
    self.fc_N6 = nn.Linear(params.hidden, params.hidden)
    self.fc_N7 = nn.Linear(params.hidden, params.hidden)
    self.fc_N8 = nn.Linear(params.hidden, params.hidden)
    self.fc_N9 = nn.Linear(params.hidden, params.hidden)

    
#     fully_connected_neurons1=50 
#     fully_connected_neurons2=50
#     fully_connected_neurons3=25
#     self.fc1 = nn.Linear(params.hidden, fully_connected_neurons1)
#     self.fc2 = nn.Linear(fully_connected_neurons1, fully_connected_neurons2)
#     self.fc4 = nn.Linear(fully_connected_neurons2, fully_connected_neurons3)    
#     self.fc3 = nn.Linear(fully_connected_neurons3, params.number_of_labels_types)
        
       #Look at this as it didn't let the NN learn on some tests 
    gain = nn.init.calculate_gain(activation_string) # gain=sqrt[2] for ReLU
#     print('gain=',gain)
#     print('---'*20)
#     print('---'*20)
  


# 
# #   ##--------------------------------------------------
#     nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
#     nn.init.orthogonal_(self.fc_h.weight, gain=gain) 
#        
#     nn.init.xavier_uniform_(self.fc_N0.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N1.weight, gain=gain) 
#     nn.init.xavier_uniform_(self.fc_N2.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N3.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N4.weight, gain=gain)     
#     nn.init.xavier_uniform_(self.fc_N5.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N6.weight, gain=gain)


#   ##-----------------------------------------------------
    nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
    nn.init.orthogonal_(self.fc_h.weight, gain=gain) 


    nn.init.xavier_uniform_(self.fc_u_inner1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_u_outer1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_u_inner.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_u_outer.weight, gain=gain)
#     nn.init.orthogonal_(self.fc_h_inner.weight, gain=gain) 
#     nn.init.orthogonal_(self.fc_h_outer.weight, gain=gain) 


       
    nn.init.orthogonal_(self.fc_N0.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N1.weight, gain=gain) 
    nn.init.orthogonal_(self.fc_N2.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N3.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N4.weight, gain=gain)     
    nn.init.orthogonal_(self.fc_N5.weight, gain=gain)

    nn.init.orthogonal_(self.fc_N6.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N7.weight, gain=gain)
    nn.init.orthogonal_(self.fc_N8.weight, gain=gain) 
    nn.init.orthogonal_(self.fc_N9.weight, gain=gain) 


#     nn.init.xavier_uniform_(self.fc_N6.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N7.weight, gain=gain)
#     nn.init.xavier_uniform_(self.fc_N8.weight, gain=gain) 
#     nn.init.xavier_uniform_(self.fc_N9.weight, gain=gain) 

# 
# #   ##-----------------------------------------------------
#     nn.init.eye_(self.fc_u.weight)
#     nn.init.eye_(self.fc_h.weight) 
#        
#     nn.init.orthogonal_(self.fc_N0.weight, gain=gain)
#     nn.init.orthogonal_(self.fc_N1.weight, gain=gain) 
#     nn.init.orthogonal_(self.fc_N2.weight, gain=gain)
#     nn.init.eye_(self.fc_N3.weight)
#     nn.init.orthogonal_(self.fc_N4.weight, gain=gain)     
#     nn.init.eye_(self.fc_N5.weight)
#     nn.init.orthogonal_(self.fc_N6.weight, gain=gain)
# 
#   ##---------------------------------------------------
#     nn.init.xavier_normal_(self.fc_u.weight, gain=gain)
# #     nn.init.orthogonal_(self.fc_h.weight, gain=gain) 
#     
#     nn.init.xavier_normal_(self.fc_h.weight, gain=gain)
#     
#     nn.init.xavier_normal_(self.fc_N1.weight, gain=gain)
#     nn.init.xavier_normal_(self.fc_N2.weight, gain=gain)
#     nn.init.xavier_normal_(self.fc_N3.weight, gain=gain)
#     nn.init.xavier_normal_(self.fc_N4.weight, gain=gain)     
#     nn.init.xavier_normal_(self.fc_N5.weight, gain=gain)
#     nn.init.xavier_normal_(self.fc_N6.weight, gain=gain) 


  ##--------------------------------------------------
  def forward(self, params, levels, children, n_inners, contents, n_level):

    n_levels = len(levels)
    embeddings = []
    
#     embeddings = torch.DoubleTensor([])
#     print('WWW'*20)
    #invert the levels using pytorch (I only invert the levels) - levels has the location of the nodes in the content list ("level by level")
    inv_idx = torch.arange(levels.size(0)-1, -1, -1).long()
    inv_levels = levels[inv_idx]     
      
    # loop over each level, starting from the bottom
    for i, nodes in enumerate(inv_levels): #So for i=0 we get the last level (leaves) =inv_levels[0]
      j = n_levels - 1 - i
      try:
          inner = nodes[:n_inners[j]]
      except ValueError:
          inner = []
      try:
          outer = nodes[n_inners[j]:n_level[j]]
      except ValueError:
          outer = []
          
      #-------------------
      # SPLIT outer vs inner NODE
#       print('j=',j)
#       print('contents[j]=',contents[j])

#       u_k_inner=torch.FloatTensor([])
#       u_k_outer=torch.FloatTensor([])
      
#       if torch.cuda.is_available():
#         u_k_inner=u_k_inner.cuda()
#         u_k_outer=u_k_outer.cuda()
        
        
        
        
      if len(inner) > 0:
        u_k_inner = self.fc_u_inner(contents[j][:n_inners[j]])
#         print('n_inners[j]=',n_inners[j])
#         print('contents[:n_inners[j]]=',contents[j][:n_inners[j]])
#         print('inner=',inner)
        
      if len(outer)>0:
#         print('outer=',outer)
        u_k_outer = self.fc_u_outer(contents[j][n_inners[j]:n_level[j]])
#         print('u_k_outer shape=',u_k_outer.shape)
#       print('---'*20)
      
      if len(inner) > 0 and len(outer)>0:
        u_k=torch.cat((u_k_inner, u_k_outer), 0)
      elif len(inner) > 0:
        u_k=u_k_inner
      else:
        u_k=u_k_outer
        
    

#  
#       #-----------------------
#       # Same inner outer weights          
#       u_k = self.fc_u(contents[j])
      
      
#       print('u_k=',u_k.shape)
#       print('===='*20) 
      
#       if j==6:
#         sys.exit()
      '''
          #Change the activation from RELU to 1X1 Convolution#
                '''
      u_k = self.activation(u_k)
  
      # implement the recursion  
      if len(inner) > 0:
          zero = torch.zeros(1).long(); one = torch.ones(1).long()
          if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
          
#           embeddings = torch.DoubleTensor(embeddings)
#           if i==0:
#             print('embeddings=',embeddings)
#             print('WWW'*20)
          
          
          h_L = embeddings[-1][children[inner, zero]]
          h_R = embeddings[-1][children[inner, one]]
        
          h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
          
#           if j==1:         
#             print('h_L=',h_L)
#             print('h=',h)
          
          h = self.fc_h(h)
          '''
              #Change the activation from RELU to 1X1 Convolution#
              '''
          h = self.activation(h)

          try:
              embeddings.append(torch.cat((h, u_k[n_inners[j]:n_level[j]]), 0))
          except ValueError:
              embeddings.append(h)
      
      else:
        embeddings.append(u_k)

#       embeddings = torch.DoubleTensor(embeddings)
#       print('embeddings[-1]=',embeddings[-1])
#       print('WWW'*20)


    
      #-----------
      #NiN layers after getting the embeddings
      
      
      h=embeddings[-1]
      
#       if j==0 :
#         h = self.fc_N2(h) 
#         
#         
#         h = self.activation(h)
#         h = self.fc_N2(h)

#       if j==0:
#         h = self.fc_N2(h)
#         
#         h = self.fc_N9(h)
      
#       
#       #---------------------------------
#       # Share weights for j=1  
#       if j==1:
#         h = self.fc_N3(h) 
#         
# #         h = F.tanh(h)
#         h = self.activation(h)
#         h = self.fc_N7(h)
#    
# #             print('j=',j)
# #             print('children[inner, zero]=',children[inner, zero])
# 
# #       print('inv_levels length=',len(inv_levels))



      ###----------------------------------------------
      # Split left/right nodes      
      if j==1:
        
        # We get the inner nodes of level j=0. So we want inv_levels[i+1]=inv_levels[n_levels-j]. This is just the list of all the tree roots
        next_inner = inv_levels[n_levels-j][:n_inners[j-1]]   
        
               
#         print('next_inner=',next_inner) 
#         print('len(next_inner)=',len(next_inner))        
#         print('children[next_inner, zero]=',children[next_inner, zero])
#         print('children[next_inner, one]=',children[next_inner, one])
#         print('h.size(0)=',h.size(0))
#         print('len(h)=',len(h))
#         print('len(children[next_inner, zero])=',len(children[next_inner, zero]))
#         print('len(children[next_inner, one])=',len(children[next_inner, one]))
        
    
        # We find the position on h (h has the embeddings for the whole minibatch) of the left (right) children of the root nodes
        left_idx = children[next_inner, zero]
        right_idx = children[next_inner, one]
        
          ##  left_idx = torch.arange(0,h.size(0),2).long()
          ##  right_idx = torch.arange(1,h.size(0),2).long()
        
#         print('left_idx=',left_idx)
#         print('right_idx=',right_idx)            
#         print('h=',h)
#         print('h[0]=',h[left_idx])
#         print('right_idx[-1]=',right_idx[-1])
#         print('h[1][-1]=',h[right_idx[-1]])
#         print('h[1]=',h[right_idx])
#         print('Number of nodes=',len(h))
# 
#         # We add a different fully connected layer for the left and right embeddings
#         h_1 = self.fc_N1(h[left_idx])  
#         h_1 = self.activation(h_1)
#         h_1 = self.fc_N2(h_1)      
#         
#         
#         h_2 = self.fc_N1(h[right_idx])  
#         h_2 = self.activation(h_2)
#         h_2 = self.fc_N2(h_2)        
#          
# #         print('h_1=',h_1)
# #         print('---'*20)
# #         print('h_2=',h_2)
        
        # We add the final embeddings for level j=1
        
        #-----------
        #New dec 9
#         print('h before=',h)
#         print('---'*20)
        h_1 = h[left_idx]
        h_2 = h[right_idx]
               
        h_1 = self.fc_N3(h_1)  
        h_1 = self.activation(h_1)
        h_1 = self.fc_N7(h_1) 

        h_2 = self.fc_N4(h_2) 
        h_2 = self.activation(h_2)
        h_2 = self.fc_N8(h_2)
        
        h_new=h
        h_new[left_idx]=h_1
        h_new[right_idx]=h_2
        
#         print('h_new after =',h_new)
#         print('---'*20)
#                 
#         h=torch.cat((h_1,h_2),0)
#         
#         print('h after 1=',h)
#         print('---'*20)       
#         
#         h = torch.reshape(h, (-1,params.hidden))
#         
#         print('h after 2=',h)
#         print('---'*20)
#         print('---'*20)
        h=h_new

                 
       

      ###--------------------------------------------
      elif j>1:
#         h = self.fc_N5(h)
#         h = self.activation(h)
#         h = self.fc_N6(h)      

# 
# #       #-----------------
#         SPLIT LEFT/RIGHT NODE
# #         if len(inner) > 0:
# #         New for order so that biggest number of constituents on the left node
#         zero = torch.zeros(1).long(); one = torch.ones(1).long()
#         # We get the inner nodes of level j-1. j decreases, so j-1 is the upper level
#         # We want the inner nodes of the level above. So we want inv_levels[i+1]=inv_levels[n_levels-j]
#         next_inner = inv_levels[n_levels-j][:n_inners[j-1]] 
#         left_idx = children[next_inner, zero]
#         right_idx = children[next_inner, one]
#       
#         h_1 = h[left_idx]
#         h_2 = h[right_idx]
# 
#         h_1 = self.fc_N2(h_1)  
# #         h_1 = self.activation(h_1)
#         h_1 = self.fc_N6(h_1) 
# 
#         h_2 = self.fc_N5(h_2) 
#         h_2 = self.fc_N9(h_2)
#       
#         h_new=h
#         h_new[left_idx]=h_1
#         h_new[right_idx]=h_2
#     
#         h=h_new



#       #-------------------
        ## SPLIT outer vs inner NODE
        if len(inner) > 0:
          zero = torch.zeros(1).long(); one = torch.ones(1).long()
          
          h_inner = h[0:n_inners[j]]
          h_1 = self.fc_N2(h_inner) 
#           h_1 = F.tanh(h_1) 
          h_1 = self.activation(h_1)
          h_1 = self.fc_N6(h_1)                
          
          h_outer = h[n_inners[j]:n_level[j]]
          if len(h_outer)>0:
            h_2 = self.fc_N5(h_outer) 
#             h_2 = F.tanh(h_2) 
            h_2 = self.activation(h_2)
            h_2 = self.fc_N9(h_2)
    
            final_h=torch.cat((h_1, h_2), 0)
  
          else:
            final_h=h_1
          
          
          h=final_h
          
#           h_new=h
#           h_new[0:n_inners[j]]=h_1
#           h_new[n_inners[j]:n_level[j]]=h_2
#           
#           print('h_new=',h_new)
#           print('---'*20)
#           print('final_h=',final_h)
#           print('---'*20)
#           print('---'*20)
#     
#           h=h_new
        
        # If there are no inners
        else:
        
          h = self.fc_N5(h) 
#           h = F.tanh(h) 
          h = self.activation(h)
          h = self.fc_N9(h)
         
          
#       #----------------------------
#       # j>=1 / 1 layer / 1 weight
#       if j>=1:
#       
#         h = self.fc_N5(h)
# 
#         h = self.activation(h)
        
        
#       h = self.fc_N5(h)
#       h = self.activation(h) 
#              
#       h = self.fc_N6(h)       
    
      
      h = self.activation(h)         


      embeddings[-1]=h
          #---------
                
#           try:
#               embeddings.append(torch.cat((h, u_k[n_inners[j]:n_level[j]]), 0))
#           except ValueError:
#               embeddings.append(h)
#       
#       else:
#         embeddings.append(u_k)

#     print('Length Embeddings=',len(embeddings))
#     print('embeddings[-1]=',embeddings[-1])
    
    
#     print('j=',j)
#     print('---'*20)
    
    h_out=embeddings[-1].view((params.batch_size, -1))
#     print('---'*20)
#     print('---'*20)  
#     print('---'*20)
#     print('---'*20)    
          
    return h_out

# ----------------------------------------
# Classifier

class PredictFromParticleEmbeddingNiN(GRNNTransformSimpleNiN): #We call GRNNTransformSimple as the superclass

  def __init__(self, params, make_embedding=None, features=None, hidden=None,**kwargs):
    super().__init__(params, features, hidden,**kwargs) #particle_transform is the RecNN architecture, e.g. GRNNTransformSimple. We pass the arguments of the __init__ of the GRNNTransformSimpleNetworking as input
  
    self.transform = make_embedding(params, features=features, hidden=hidden,**kwargs) 

    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc1 = nn.Linear(params.hidden, params.hidden)
    self.fc2 = nn.Linear(params.hidden, params.hidden)
    self.fc3 = nn.Linear(params.hidden, params.number_of_labels_types)

    gain = nn.init.calculate_gain(activation_string)
    
   # ---------------------------------------- 
    nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc3.weight, gain=gain)
    nn.init.constant_(self.fc3.bias, 1)   
     
# 
#    # -------------------------------------- 
#     nn.init.orthogonal_(self.fc1.weight, gain=gain)
#     nn.init.orthogonal_(self.fc2.weight, gain=gain)
#     nn.init.orthogonal_(self.fc3.weight, gain=gain)
# #     nn.init.constant_(self.fc3.bias, 1) 
# 
#    # -------------------------------------- 
#     nn.init.eye_(self.fc1.weight)
#     nn.init.eye_(self.fc2.weight)
#     nn.init.eye_(self.fc3.weight)
# #     nn.init.constant_(self.fc3.bias, 1) 

       
#     activation_string = 'relu'
#     self.activation = getattr(F, activation_string)

  ##-------------------------------------------------
#   def forward(self, params, levels, children, n_inners, contents, n_level):

  def forward(self, params, levels, children, n_inners, contents, n_level, **kwargs):
    h_out = self.transform(params, levels, children, n_inners, contents, n_level, **kwargs)

    ##-------------------
    #concatenate a fully-connected NN
    h_out = self.fc1(h_out)

    h_out = self.activation(h_out)

    h_out = self.fc2(h_out)

    h_out = self.activation(h_out)


  #     h_out = self.fc4(h_out)  
  #     h_out = self.activation(h_out)

  #     h_out = F.log_softmax(self.fc3(h_out), dim=1) # dim: batch_size*seq_len x num_tags

    output = torch.sigmoid(self.fc3(h_out))
    # output = F.sigmoid(self.fc3(h_out))


    return output 





#-------------------------------------------------------------------------------------------------------------
# 2) Simple RecNN
#-------------------------------------------------------------------------------------------------------------
# Make embedding
class GRNNTransformSimple(nn.Module):
  '''
  Recursive neural network architecture. nn.Module is the NN superclass in torch. First creates the recursive graph and then concatenates a fully-connected NN that ends with 2 nodes (binary classification).
  '''

  def __init__(self, params, features=None, hidden=None,**kwargs):
    super(GRNNTransformSimple,self).__init__()
        
    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc_u = nn.Linear(params.features, params.hidden)
    self.fc_h = nn.Linear(3 * params.hidden, params.hidden) 


#     fully_connected_neurons1=50 
#     fully_connected_neurons2=50
#     fully_connected_neurons3=25
#     self.fc1 = nn.Linear(params.hidden, fully_connected_neurons1)
#     self.fc2 = nn.Linear(fully_connected_neurons1, fully_connected_neurons2)
#     self.fc4 = nn.Linear(fully_connected_neurons2, fully_connected_neurons3)    
#     self.fc3 = nn.Linear(fully_connected_neurons3, params.number_of_labels_types)
        
       #Look at this as it didn't let the NN learn on some tests 
    gain = nn.init.calculate_gain(activation_string)
    nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
    nn.init.orthogonal_(self.fc_h.weight, gain=gain) 
                
        
#     activation_string = 'relu'
#     self.activation = getattr(F, activation_string)

  ##-----------------------------------------------
  def forward(self, params, levels, children, n_inners, contents, n_level):

    n_levels = len(levels)
    embeddings = []
   
    #invert the levels using pytorch
    inv_idx = torch.arange(levels.size(0)-1, -1, -1).long()
    inv_levels = levels[inv_idx]     
      
    # loop over each level, starting from the bottom
    for i, nodes in enumerate(inv_levels):
      j = n_levels - 1 - i
      try:
          inner = nodes[:n_inners[j]]
      except ValueError:
          inner = []
      try:
          outer = nodes[n_inners[j]:n_level[j]]
      except ValueError:
          outer = []
      u_k = self.fc_u(contents[j])
      '''
          #Change the activation from RELU to 1X1 Convolution#
                '''
      u_k = self.activation(u_k)
  
      # implement the recursion  
      if len(inner) > 0:
          zero = torch.zeros(1).long(); one = torch.ones(1).long()
          if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
          h_L = embeddings[-1][children[inner, zero]]
          h_R = embeddings[-1][children[inner, one]]
        
          h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
          h = self.fc_h(h)
          '''
              #Change the activation from RELU to 1X1 Convolution#
              '''
          h = self.activation(h)
                
          try:
              embeddings.append(torch.cat((h, u_k[n_inners[j]:n_level[j]]), 0))
          except ValueError:
              embeddings.append(h)
      
      else:
        embeddings.append(u_k)

#     print('Length Embeddings=',len(embeddings))
#     print('embeddings[-1]=',embeddings[-1])
    
    h_out=embeddings[-1].view((params.batch_size, -1))
        
    return h_out

# --------------------------------------
# Classifier

class PredictFromParticleEmbedding(GRNNTransformSimple): #We call GRNNTransformSimple as the superclass

  def __init__(self, params, make_embedding=None, features=None, hidden=None,**kwargs):
    super().__init__(params, features, hidden,**kwargs) #particle_transform is the RecNN architecture, e.g. GRNNTransformSimple. We pass the arguments of the __init__ of the GRNNTransformSimple as input
  
    self.transform = make_embedding(params, features=features, hidden=hidden,**kwargs) 

    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc1 = nn.Linear(params.hidden, params.hidden)
    self.fc2 = nn.Linear(params.hidden, params.hidden)
    self.fc3 = nn.Linear(params.hidden, params.number_of_labels_types)

    gain = nn.init.calculate_gain(activation_string)
    nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc3.weight, gain=gain)
    nn.init.constant_(self.fc3.bias, 1)        
        
#     activation_string = 'relu'
#     self.activation = getattr(F, activation_string)

  ##-------------------------------------------------
#   def forward(self, params, levels, children, n_inners, contents, n_level):

  def forward(self, params, levels, children, n_inners, contents, n_level, **kwargs):
    h_out = self.transform(params, levels, children, n_inners, contents, n_level, **kwargs)

    ##-------------------
    #concatenate a fully-connected NN
    h_out = self.fc1(h_out)

    h_out = self.activation(h_out)

    h_out = self.fc2(h_out)

    h_out = self.activation(h_out)


  #     h_out = self.fc4(h_out)  
  #     h_out = self.activation(h_out)

  #     h_out = F.log_softmax(self.fc3(h_out), dim=1) # dim: batch_size*seq_len x num_tags

    output = torch.sigmoid(self.fc3(h_out))
    # output = F.sigmoid(self.fc3(h_out))


    return output 



#-------------------------------------------------------------------------------------------------------------
# 3) Gated RecNN
#-------------------------------------------------------------------------------------------------------------
# Make embedding
class GRNNTransformGated(nn.Module):
  '''
  Recursive neural network architecture. nn.Module is the NN superclass in torch. First creates the recursive graph and then concatenates a fully-connected NN that ends with 2 nodes (binary classification).
  '''

  def __init__(self, params, features=None, hidden=None,**kwargs):
    super().__init__()
        
#     self.iters = iters
    
    activation_string = 'relu' 
    self.activation = getattr(F, activation_string)

    self.fc_u = nn.Linear(params.features, params.hidden)
    self.fc_h = nn.Linear(3 * params.hidden, params.hidden) 
    self.fc_z = nn.Linear(4 * params.hidden, 4 * params.hidden)
    self.fc_r = nn.Linear(3 * params.hidden, 3 * params.hidden)

        
       #Look at this as it didn't let the NN learn on some tests 
    gain = nn.init.calculate_gain(activation_string)
    nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
    nn.init.orthogonal_(self.fc_h.weight, gain=gain) 
    nn.init.xavier_uniform_(self.fc_z.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc_r.weight, gain=gain)                
        
#     if self.iters > 0:
#         self.down_root = nn.Linear(params.hidden, params.hidden)
#         self.down_gru = AnyBatchGRUCell(params.hidden, params.hidden)

  #------------------------------------------------
  def forward(self, params, levels, children, n_inners, contents, n_level):
  
    n_levels = len(levels)
    up_embeddings = [None for _ in range(n_levels)]
#     down_embeddings = [None for _ in range(n_levels)]
    
    
#     hidden = self.hidden
#     conv = False
    
    #invert the levels using pytorch
    inv_idx = torch.arange(levels.size(0)-1, -1, -1).long()
    inv_levels = levels[inv_idx]     
      
    # loop over each level, starting from the bottom
    for i, nodes in enumerate(inv_levels):
      j = n_levels - 1 - i
      try:
          inner = nodes[:n_inners[j]]
      except ValueError:
          inner = []
      try:
          outer = nodes[n_inners[j]:n_level[j]]
      except ValueError:
          outer = []
      u_k = self.fc_u(contents[j])

      u_k = self.activation(u_k)
 

      if len(inner) > 0:
          zero = torch.zeros(1).long(); one = torch.ones(1).long()
          if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
          h_L = up_embeddings[j+1][children[inner, zero]]
          h_R = up_embeddings[j+1][children[inner, one]]
        
          hhu = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
          r = self.fc_r(hhu)  #Eq. (A6)
#           r = F.sigmoid(r)          
          
          r = torch.sigmoid(r)
          
          h_H = self.fc_h(r * hhu)  #Eq. (A4)
          h_H = self.activation(h_H)


          z = self.fc_z(torch.cat((h_H, hhu), -1)) #Eq (A5)
          
          z_H = z[:, :params.hidden]               # new activation
          z_L = z[:, params.hidden:2*params.hidden]     # left activation
          z_R = z[:, 2*params.hidden:3*params.hidden]   # right activation
          z_N = z[:, 3*params.hidden:]             # local state
          
          z = torch.stack([z_H,z_L,z_R,z_N], 2)
#           z = F.softmax(z)

          z = F.softmax(z, dim=1)

          h = ((z[:, :, 0] * h_H) +
               (z[:, :, 1] * h_L) +
               (z[:, :, 2] * h_R) +
               (z[:, :, 3] * u_k[:n_inners[j]])) #Eq (A1)
               

          try:
            up_embeddings[j]= torch.cat((h, u_k[n_inners[j]:n_level[j]]), 0)
          except AttributeError:
            up_embeddings[j] = h
      
      else:
        up_embeddings[j] = u_k
 


    h_out=up_embeddings[0].view((params.batch_size, -1))
        
    return h_out
    
    
#   ##-------------------------------------------------
# Classifier

class PredictFromParticleEmbeddingGated(GRNNTransformGated): #We call GRNNTransformSimple as the superclass

  def __init__(self, params, make_embedding=None, features=None, hidden=None,**kwargs):
    super().__init__(params, features, hidden,**kwargs) #particle_transform is the RecNN architecture, e.g. GRNNTransformSimple. We pass the arguments of the __init__ of the GRNNTransformSimple as input
  
    self.transform = make_embedding(params, features=features, hidden=hidden,**kwargs) 

    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc1 = nn.Linear(params.hidden, params.hidden)
    self.fc2 = nn.Linear(params.hidden, params.hidden)
    self.fc3 = nn.Linear(params.hidden, params.number_of_labels_types)

    gain = nn.init.calculate_gain(activation_string)
    nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc3.weight, gain=gain)
    nn.init.constant_(self.fc3.bias, 1)        
        
#     activation_string = 'relu'
#     self.activation = getattr(F, activation_string)

  ##-------------------------------------------------
#   def forward(self, params, levels, children, n_inners, contents, n_level):

  def forward(self, params, levels, children, n_inners, contents, n_level, **kwargs):
    h_out = self.transform(params, levels, children, n_inners, contents, n_level, **kwargs)

    ##-------------------
    #concatenate a fully-connected NN
    h_out = self.fc1(h_out)

    h_out = self.activation(h_out)

    h_out = self.fc2(h_out)

    h_out = self.activation(h_out)


  #     h_out = self.fc4(h_out)  
  #     h_out = self.activation(h_out)

  #     h_out = F.log_softmax(self.fc3(h_out), dim=1) # dim: batch_size*seq_len x num_tags

    output = torch.sigmoid(self.fc3(h_out))
    # output = F.sigmoid(self.fc3(h_out))


    return output 




    
 
            
##############################################################################################################
#/////////////////////      OTHER FUNCTIONS     //////////////////////////////////////////////////////////////
##############################################################################################################
# compute the accuracy    
def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. 
    Args:
        outputs: (np.ndarray) dimension batch_size x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size  where each element is a label in
                [0, 1, ... num_tag-1]
    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()
#     print('labels for accuracy=',labels)
    
    ##-----------------------------------------------
    # np.argmax gives us the class predicted for each token by the model
#     print('outputs before argmax=',outputs)

    #If more than one output class
#     outputs = np.argmax(outputs, axis=1)
    
    #If one output neuron
    outputs=np.rint(outputs)
    
#     print('outputs after argmax=',outputs)


    outputs=np.asarray(outputs)
    labels=np.asarray(labels)
    
    outputs=outputs.flatten()
    ##-----------------------------------------------
    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
#     print('np.sum(outputs==labels)=',np.sum(outputs==labels))
#     print('labels=',labels)
#     print('outputs=',outputs)
    return np.sum(outputs==labels)/float(len(labels))

#-------------------------------------------------------------------------------------------------------------
# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}