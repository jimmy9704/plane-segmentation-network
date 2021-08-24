import numpy as np
import math
import random
import scipy.spatial.distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import open3d as o3d
#from open3d.open3d.geometry import voxel_down_sample,estimate_normals
from copy import deepcopy
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return 1 - dot(A, B)/(norm(A)*norm(B))
def distance(a, b):
    return sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))]) ** 0.5

def Kmeans_n(inputs,k,rate):
  random_inputs = np.random.permutation(inputs)

  #normal estimation
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(random_inputs)
  #N = estimate_normals(pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
  N = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
  normals = np.asarray(pcd.normals)
  normals = random_inputs/abs(random_inputs)*abs(normals)

  input_points = np.hstack([random_inputs,normals])

  #kmeans++ centroid
  X =KMeans(n_clusters=k,n_init=1,max_iter=1).fit(input_points)
  centroids = X.cluster_centers_

  centroids_old = np.zeros(centroids.shape)
  labels = np.zeros(len(input_points))
  error = np.zeros(k)
  for i in range(k):
    error[i] = distance(centroids_old[i,:3], centroids[i,:3])
  #while(np.mean(error)>0.001):
  for i in range(25):
    if (np.mean(error)<0.001):
      break;
    for i in range(len(input_points)):
      distances = np.zeros(k)
      for j in range(k):
        distances[j] =  1*distance(input_points[i,:3], centroids[j,:3]) + rate*cos_sim(input_points[i,3:], centroids[j,3:])
      cluster = np.argmin(distances)
      labels[i] = cluster
    centroids_old = deepcopy(centroids)
    cos_losses = np.zeros(k)
    for i in range(k):
      points = [ input_points[j] for j in range(len(input_points)) if labels[j] == i ]
      points = np.asarray(points)
      centroids[i] = np.mean(points, axis=0)
      for j in range(len(points)):
        cos_losses[i] += cos_sim(points[j,3:],centroids[i,3:])
      cos_losses[i] = cos_losses[i]/len(points)
    cos_loss = np.max(cos_losses[~np.isnan(cos_losses)])

    for i in range(k):
      error[i] = distance(centroids_old[i], centroids[i])
  return labels,input_points,cos_loss, centroids

def Kmeans_normal(inputs,k,iter=10):
  labels_arr = np.zeros((iter,len(inputs)))
  outputs_arr = np.zeros((iter,len(inputs),6))
  cos_loss_arr = np.zeros((iter))
  centroids_arr = np.zeros((iter,k,6))
  for i in range(iter):
    labels_arr[i],outputs_arr[i], cos_loss_arr[i], centroids_arr[i]= Kmeans_n(inputs,k,60)
  idx = np.nanargmin(cos_loss_arr)
  cos_loss = cos_loss_arr[idx]
  labels = labels_arr[idx]
  outputs = outputs_arr[idx]
  centroids = centroids_arr[idx]
  return labels,outputs,cos_loss, centroids

def Kmeans(inputs,k,iter=10):
  labels_arr = np.zeros((iter,len(inputs)))
  outputs_arr = np.zeros((iter,len(inputs),6))
  cos_loss_arr = np.zeros((iter))
  centroids_arr = np.zeros((iter,k,6))
  for i in range(iter):
    labels_arr[i],outputs_arr[i], cos_loss_arr[i], centroids_arr[i]= Kmeans_n(inputs,k,0)
  #cos_loss_arr[np.isnan(cos_loss_arr)] = 1
  idx = np.nanargmin(cos_loss_arr)
  cos_loss = cos_loss_arr[idx]
  labels = labels_arr[idx]
  outputs = outputs_arr[idx]
  centroids = centroids_arr[idx]
  return labels,outputs,cos_loss, centroids


