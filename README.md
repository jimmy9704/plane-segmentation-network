# Clustering Based Plane Segmentation Network for Urban Scene Modeling
## Network Architecture
![network](./images/HKPS.png)

## Results
![results](./images/results.jpg)

## Requirements
* Python 3.8
* Pytorch 1.2

## Installation
```
git clone https://github.com/jimmy9704/plane-segmentation-network.git
cd plane-segmentation-network/
```

## Usage
you can use the notebook HKPS.ipynb to train and evaluate HKPS(Hybrid K-means Plane Segmentation Network)
 
## Training
```
train(pointnet, dataset,train_loss, epochs=100,make_label=False, save=True)
```
you can change 'max_k' and 'iteration'


'max_k' is the maximum number that PointNet will estimate


'iteration' is the number of Hybrid-Kmeans iterations


'iteration' can be reduced to reduce time consumption but it might cause unstable results

## Evaluation
```
python eval.py
```

