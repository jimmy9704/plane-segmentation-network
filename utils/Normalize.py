import numpy as np

def normalize(dataset_):
    dataset = dataset_
    xmin = []
    xmax = []
    for i in range(3):
        xmin.append(np.min(dataset[:,:,i]))
        xmax.append(np.max(dataset[:,:,i]))

    print(xmin)
    print(xmax)

    dataset_normal = np.zeros(dataset.shape)
    for i in range(3):
        dataset_normal[:,:,i] = (dataset[:,:,i] - xmin[i]) / (xmax[i] - xmin[i])

    features = ["x","y","z"]
    for i in range(3): 
        print(features[i] + "_range :", np.min(dataset_normal[:, :, i]), np.max(dataset_normal[:, :, i]))
        
    return dataset