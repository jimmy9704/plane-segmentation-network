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


def valid(model, dataset,save_num=99,iteration=10):
  if(len(dataset.shape) <3):
    dataset = dataset.reshape(1,1600,-1)
  #normalize dataset
  dataset_normal = Normalize.normalize(dataset)

  inputs = torch.tensor(dataset_normal).to(device).float()
  file = "./model_save/save_" + str(save_num) + ".pth"
  pointnet.load_state_dict(torch.load(file))
  print(inputs.shape)
  pointnet.eval()
  with torch.no_grad():
    outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
  outputs_ = F.softmax(outputs)
  outputs_ = torch.argmax(outputs_,dim =1)

  print("PointNet result: ", outputs_ + 1)
  K = np.asarray(outputs_.cpu()+1)
  for n in range(len(K)):
      print("%d / %d" %(n+1, len(K)))
      labels_k_,outputs_k,cos_loss,centroids = Hybrid_kmeans.Kmeans_normal(dataset[n],K[n],iter=iteration)
      labels_k = Merge.PlaneMerge(outputs_k,np.copy(labels_k_),K[n])
      result_file = "HKPS_" + str(n)
      Visualize.visualize(labels_k,outputs_k,result_file)
  print("Result Saved")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
pointnet = Pointnet.PointNet()
pointnet.to(device);
# load dataset
# The dataset is an unorganized point cloud in the form of Nx3 (xyz)
# preprocessed by voxel downsamling.
dataset = np.load("./dataset/with_noise.npy",allow_pickle=True)
print(dataset.shape)

#Results are saved as result_HKPS_'index'.txt
#The result is shape as xyzRGB
#'iteration' can be reduced to reduce time consumption
#but it might cause unstable results
valid(pointnet, dataset[:10], save_num=99,iteration=5)