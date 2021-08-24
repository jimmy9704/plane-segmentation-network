##  Nearest Plane Merge ##
import numpy as np
import Hybrid_kmeans

def Dist_dataset(dataset1, dataset2):
  for i in range(len(dataset1)):
    for j in range(len(dataset2)):
      if (Hybrid_kmeans.distance(dataset1[i,:3],dataset2[j,:3])<4):
        #print("Detect Merge Plane")
        return True
  return False

def PlaneMerge(dataset,labels, K):
  relabels = np.array(range(K))
  for re in range(K+2):
    K = int(max(labels) + 1)
    for i in range(K):
      #print(i)
      for j in range(K):
        mask_array = labels == i
        dataset1 = dataset[mask_array]
        centroids_1 = np.mean(dataset1,axis = 0)
        mask_array = labels == j
        dataset2 = dataset[mask_array]
        centroids_2 = np.mean(dataset2,axis = 0)
        if ((i != j) and (Hybrid_kmeans.cos_sim(centroids_1[3:],centroids_2[3:])<0.1)):
          #print("Detect Similar Normal")
          merge = Dist_dataset(dataset1,dataset2)
          if (merge == True):
            relabels[j] = relabels[i]
    for k in range(K):
      relabels[k] = relabels[relabels[k]]
    for i in range(len(labels)):
      labels[i] = relabels[int(labels[i])]
  print("FinishMerge")  

  count = 0
  for i in range(20):
    num = 0
    for j in range(len(labels)):
      num = j
      if (labels[j] == i):
        break;
    if (num != (len(labels)-1)):
      for j in range(len(labels)):
        if (labels[j] == i):
          labels[j] = count;
      count += 1

  return labels

