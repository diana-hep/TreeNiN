import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.autograd import Variable

##############################################################################################################
#/////////////////////   CLASSES     ////////////////////////////////////////////////////////////////////////
##############################################################################################################

#Recursive neural network architecture. 
class GRNNTransformSimple_level(nn.Module):
  '''
  Recursive neural network architecture. nn.Module is the NN superclass in torch. First creates the recursive graph and then concatenates a fully-connected NN that ends with 2 nodes (binary classification).
  '''

  def __init__(self, params, features=None, hidden=None,**kwargs):
    super(GRNNTransformSimple_level,self).__init__()
        
    activation_string = 'relu'
    self.activation = getattr(F, activation_string)

    self.fc_u = nn.Linear(params.features, params.hidden)
    self.fc_h = nn.Linear(3 * params.hidden, params.hidden) 

    self.fc1 = nn.Linear(params.hidden, params.hidden)
    self.fc2 = nn.Linear(params.hidden, params.hidden)
    self.fc3 = nn.Linear(params.hidden, params.number_of_labels_types)

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
        
    nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
    nn.init.xavier_uniform_(self.fc3.weight, gain=gain)
    nn.init.constant_(self.fc3.bias, 1)        
        
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
    
#     print('h_out=',h_out)
    
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