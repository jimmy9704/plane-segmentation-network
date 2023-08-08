import numpy as np
import math
import random
import os
import sys
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sys.path.append('./utils')
import Hybrid_kmeans
import Pointnet
import Merge
import Visualize
import Normalize

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
pointnet = Pointnet.PointNet()
pointnet.to(device);
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)

# load dataset
# The dataset is an unorganized point cloud in the form of Nx3 (xyz)
# preprocessed by voxel downsamling.
dataset = np.load("./dataset/with_noise.npy",allow_pickle=True)
print(dataset.shape)

def pointnetloss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def make_labels(dataset,max_k=15,iteration=10):
    labels = np.zeros(len(dataset))
    for size in range(len(dataset)):
      print("make labels %d / %d: " % (int(size)+1,len(dataset)))
      loss = np.zeros(max_k)
      for k in range(1,max_k+1):
        k = int(k)
        #Kmeans
        #labels_,outputs,cos_loss,centroids = Hybrid_kmeans.Kmeans(dataset[size],k,10)
        #Hybrid_Kmeans
        labels_,outputs,cos_loss,centroids = Hybrid_kmeans.Kmeans_normal(dataset[size],k,iter=iteration)
        loss[k-1] = cos_loss 
      labels[size] = max_k-1
      for i in range(max_k-1,0,-1):
        if ((loss[i-1]-loss[i]) > 0.01):
          labels[size] = i
          break
      print("label : ",labels[size]+1)
      np.save("HKPS_labels",labels)
    
def train(model,dataset_,train_loss, epochs=5,make_label=True, save=True):
    
    #you can change 'max_k' and 'iteration'
    #'max_k' is the maximum number that PointNet will estimate
    #'iteration' is the number of Hybrid-Kmeans iterations
    #'iteration' can be reduced to reduce time consumption
    #but it might cause unstable results
    if (make_label == True):
        make_labels(dataset,max_k=15,iteration=10)
    
    #normalize dataset
    dataset = Normalize.normalize(dataset_)
    
    labels_ = np.load("HKPS_labels.npy",allow_pickle=True)

    print("Labels: ",labels_ +1 )
    print("train pointnet")
    inputs_, labels_ = torch.tensor(dataset).to(device).float(), torch.tensor(labels_).to(device).long()
    for epoch in range(epochs): 
        shuf_idx = np.random.permutation(len(labels_))
        inputs_ = inputs_[shuf_idx]
        labels_ = labels_[shuf_idx]

        pointnet.train()
        batch_size = int(len(inputs_)/20)
        for i in range(batch_size):
            inputs = inputs_[20*i:20*(i+1)]
            labels = labels_[20*i:20*(i+1)]
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
            outputs_ = F.softmax(outputs)
            outputs_ = torch.argmax(outputs_,dim =1)

            loss = pointnetloss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # print statistics
            print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, batch_size, loss.item()), end='\r')

        # save the model
        if save:
          torch.save(pointnet.state_dict(), "./model_save/save_"+str(epoch)+".pth")
        
#if you want to make labels for PointNet training run make_labels or modify make_label as True
#'iteration' can be reduced to reduce time consumption
#but it might cause unstable results

train_loss = []
#make_labels(dataset,max_k=15,iteration=10)
train(pointnet, dataset,train_loss, epochs=50,make_label=False, save=True)
