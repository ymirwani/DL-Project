# Deep Learning Project: Sparsity-Invariant-CNNs
**Members of Group 37**

Yash Mirwani (4545664), 
Ravi Snellenberg (5315867),
Devin Tomur (4806190),
Gijs Zijderveld (5306728)

Link: https://arxiv.org/abs/1708.06500

The data-sets used for reproducing this project are:
1. MNIST
2. Caltech101


## Introduction
**NOTE:** This is abstract of the paper we are following:


In this paper, we consider convolutional neural networks
operating on sparse inputs with an application to depth upsampling from sparse laser scan data. First, we show that
traditional convolutional networks perform poorly when
applied to sparse data even when the location of missing
data is provided to the network. To overcome this problem,
we propose a simple yet effective sparse convolution layer
which explicitly considers the location of missing data during the convolution operation. We demonstrate the benefits
of the proposed network architecture in synthetic and real
experiments with respect to various baseline approaches.
Compared to dense baselines, the proposed sparse convolution network generalizes well to novel datasets and is invariant to the level of sparsity in the data. For our evaluation, we derive a novel dataset from the KITTI benchmark,
comprising 93k depth annotated RGB images. Our dataset
allows for training and evaluating depth upsampling and
depth prediction techniques in challenging real-world settings and will be made available upon publication

## Method

## Results and Conclusions

## Setup

Requirements:

- Linux Ubuntu (tested on versions XYZ)
- CUDA
- Python 3.12

Installation:

- Windows:

- Linux:

* `sudo apt install `

- MAC:

## References

[1] Uhrig, J., Schneider, N., Schneider, L., Franke, U., Brox, T., & Geiger, A. (2017, October). Sparsity invariant cnns. In 2017 international conference on 3D Vision (3DV) (pp. 11-20). IEEE.

[2] 


If you find our work useful in your research, please consider citing:
