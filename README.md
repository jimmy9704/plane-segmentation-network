# Clustering Based Plane Segmentation Network for Urban Scene Modeling
## Network Architecture
![network](./images/HKPS.png)

## Results
![results](./images/results.jpg)

## Requirements
* Python 3.8
* Pytorch 1.2
* scikit-learn 0.24.2
* Open3d 0.13.0

## Installation
```
git clone https://github.com/jimmy9704/plane-segmentation-network.git
cd plane-segmentation-network/
```

## Usage
you can use the notebook HKPS.ipynb to train and valid HKPS(Hybrid K-means Plane Segmentation Network)
 
## Training
```
train(pointnet, dataset,train_loss, epochs=100,make_label=False, save=True)
make_labels(dataset,max_k=15,iteration=10)
```
You can create a label for training PointNet by setting the 'make_label' option to 'True'.

You can change 'max_k' and 'iteration'.

'max_k' is the maximum number that PointNet will estimate

'iteration' is the number of Hybrid-Kmeans iterations

'iteration' can be reduced to reduce time consumption but it might cause unstable results.

## Validation
```
valid(pointnet, dataset, save_num=99,iteration=5)
```
'save_num' is the check point to load for PointNet.


Pre-trained checkpoints can be loaded by setting 'save_num' to 99.
