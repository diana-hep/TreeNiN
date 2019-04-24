# import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import random
from sklearn.utils import check_random_state


##-----------------------------------------------------------------------------------------------------------
class TreeDataset(data.Dataset):
  '''
  Subclass of the data.Dataset class. We override __len__, that provides the size of the dataset, and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
  Args:
    data: dataset
    labels: labels of each element of the dataset
    transform: function that we want to apply to the data. For trees, this will be that function that creates the training batches
    batch_size: size of the training batches
    features: Number of features in each node
  ''' 
  ##----
  def __init__(self,data=None,labels=None,shuffle=True,transform=None,batch_size=None,features=None):
  
    self.data=data
    self.labels=labels
    self.transform=transform
    self.batch_size=batch_size
    self.features=features
  
    if shuffle:   
      indices = check_random_state(seed=None).permutation(len(self.data))
#       print('self.data=',self.data[0]['tree'])
#       print('self.labels=',self.labels)
      self.data=self.data[indices]
      self.labels=self.labels[indices]
#       print('-+-+'*20)
#       print('self.data=',self.data[0]['tree'])
#       print('self.labels=',self.labels)

  ##----
  # Override __getitem__
  def __getitem__(self,index):
  
    if self.transform is not None:
      levels, children, n_inners, contents, n_level= self.transform(self.data[index*self.batch_size:(index+1)*self.batch_size],self.features)

      # Shift to np arrays
      levels = np.asarray(levels)
      children = np.asarray(children)
      n_inners = np.asarray(n_inners)
      contents = np.asarray(contents)
      n_level = np.asarray(n_level)
      labels= np.asarray(self.labels[index*self.batch_size:(index+1)*self.batch_size])

    return levels, children, n_inners, contents, n_level, labels

  ##----
  # Override __len__, that provides the size of the dataset
  def __len__(self):
    return len(self.data)


##-----------------------------------------------------------------------------------------------------------
def customized_collate(batch):
  """"
  default_collate contains definitions of the methods used by the _DataLoaderIter workers to collate samples fetched from dataset into Tensor(s).
  These **needs** to be in global scope since Py2 doesn't support serializing
  static methods.
  Here we define customized_collate that returns the elements of each batch tuple shifted to pytorch tensors.
  """  
  
  levels=torch.LongTensor(batch[0][0])
  children=torch.LongTensor(batch[0][1])
  n_inners=torch.LongTensor(batch[0][2])
  contents = torch.FloatTensor(batch[0][3])
  n_level=torch.LongTensor(batch[0][4])
  labels= torch.LongTensor(batch[0][5])
  
  return levels, children, n_inners, contents, n_level, labels








