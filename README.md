# Deep Learning Project: Sparsity-Invariant-CNNs
**Members of Group 37**

Yash Mirwani (4545664), 
Ravi Snellenberg (5315867),
Devin Tomur (4806190),
Gijs Zijderveld (5306728)

## Links

- Paper: [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)

## Datasets

The datasets used for reproducing the results of this project are:
1. [MNIST](http://yann.lecun.com/exdb/mnist/)
2. [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)


## Method


In this repository, we consider convolutional neural networks operating on sparse inputs with an application to depth upsampling from sparse laser scan data. First, we show that traditional convolutional networks perform poorly when applied to sparse data, even when the location of missing data is provided to the network. To overcome this challenge, we propose a simple yet effective sparse convolution layer that explicitly considers the location of missing data during the convolution process. We demonstrate the benefits of the proposed network architecture through synthetic and real experiments against various baseline approaches. Compared to dense baselines, our sparse convolution network generalizes well to novel datasets and is invariant to the level of sparsity in the data. For our evaluation, we derived a novel dataset from the KITTI benchmark, comprising 93k depth-annotated RGB images. This dataset allows for comprehensive training and evaluation of depth upsampling and prediction techniques under challenging real-world conditions and will be made available upon publication. 

The original CNN model was changed so the hyperparameters could easily be adjusted as arguments to a model class, the python notebook used for this can be found on the github under 'hp_optimization.ipynb'. It was quickly realized that training the model, even on Google Colab’s cloud computing, took about 30 seconds for each epoch (a full pass through the training data), which makes computing this for a lot of epochs in total and a large combination of hyperparameters unsuitable.

### Implementation details
We replicated and tested using the architecture and Sparse Convolution described in Figure 1. We did so using Python as it is the most commenly used languange for machine learning tasks. So anyone that plans on using this Sparse Covolution layer will most likely be doing so using Python. Python also offers a large ammount of integrated support for creating nice visualizations which is good to demonstrate our results. 

Using Python does come with a few caveats i.e. it is inherently a very slow language that just isn't suited for machine learning. The way Python gets around this is by implementing the often used modules(such as NumPy) in C++ and referencing that code. This generally works quite well but it restricts us to using existing modules if we want a somewhat efficient implementation. Using existing modules also makes the implementation more effiecient for the GPU as existing modules are genneraly already optimized for that.

![ima![hp_results](https://hackmd.io/_uploads/HJLdrdteA.png)
ge](https://hackmd.io/_uploads/Bkwaho9yR.png)
Figure 1 (From: https://arxiv.org/abs/1708.06500v2)

### Blog

We created a blog that gives more information about this project:

https://hackmd.io/@IsOgP4nuSI6amqHGvP2KjA/rkQL5RMyC 

## Setup

### Requirements

- Linux Ubuntu (tested on versions 18.04, 20.04, and 22.04)
- Python 3.12

### Cloning the Repository

```bash
git clone https://github.com/ymirwani/DL-Project.git
```

### References

[1] Uhrig, J., Schneider, N., Schneider, L., Franke, U., Brox, T., & Geiger, A. (2017, October). Sparsity invariant cnns. In 2017 international conference on 3D Vision (3DV) (pp. 11-20). IEEE.
