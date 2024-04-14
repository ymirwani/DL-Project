Authors:
*  Creation of new Sparsity CNN + testing / validation on the MNIST dataset + comparison with default CNN with similar layer structure:
    * Ravi Snellenberg(student number: 5315867, contact: R.snellenberg@student.tudelft.nl)
    * Yash Mirwani(student number: 4545664, contact: y.m.mirwani@student.tudelft.nl)
* Hyperparameter tuning:
    * Devin Tomur (student number: 4806190, contact: d.tomur@student.tudelft.nl)
* New sparsity CNN vs default CNN on Caltech-101 dataset:
    * Gijs Zijderveld(student number: 5306728, contact G.J.Zijderveld@student.tudelft.nl)

Github Code: https://github.com/ymirwani/DL-Project
Datasets:
* MNIST
* Caltech101
## Introduction
Convolutional neural networks (CNNs) have transformed our ability to interpret and understand complex, high-dimensional data, particularly images. Despite their success, traditional CNNs often falter when faced with sparse inputs - a scenario frequently encountered in fields like autonomous driving where depth sensors provide critical but sparsely distributed information about the environment. Recognizing this challenge, the pioneering research conducted by Jonas Uhrig, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and Andreas Geiger, presented in their paper "Sparsity Invariant CNNs," introduces a novel approach that significantly enhances the efficacy of CNNs in dealing with sparse data.

The key innovation introduced by Uhrig et al. is the development of a sparse convolutional layer engineered to process sparse inputs efficiently. This layer modifies the standard convolution operation to be insensitive to data sparsity ensuring consistent performance across varying levels of input completeness. This innovation holds significant potential for robotics and autonomous vehicle applications where sensor data may be inherently sparse or unevenly distributed across the observed environment.

Here, we implement these concepts from scratch using PyTorch. This blog post is an exploration of our experience implementing the sparsity invariant CNN proposed by Uhrig et al., the challenges encountered along the way, and the insights gained from applying this innovative approach to different datasets. 



## New sparsity CNN from the ground up
In this section, we will go over our attempt at reproducing the sparse CNN described in the paper. This reproduction was done without looking at any existing implementation of a sparse CNN and was solely based on what was described in the paper. To see if the paper gives sufficient information to fully reproduce its results, we fully implement it from scratch(without significant extra work done).  

### Implementation details
#### Setup
We replicated and tested using the architecture and Sparse Convolution described in Figure 1. We did so using Python as it is the most commonly used language for machine learning tasks. So anyone that plans on using this Sparse Convolution layer will most likely be doing so using Python. Python also offers a large amount of integrated support for creating nice visualizations which is good for demonstrating our results. 

Using Python does come with a few caveats i.e. it is inherently a very slow language that just isn't suited for machine learning. The way Python gets around this is by implementing the often-used modules(such as NumPy) in C++ and referencing that code. This generally works quite well but it restricts us to using existing modules if we want a somewhat efficient implementation. Using existing modules also makes the implementation more efficient for the GPU as existing modules are generally already optimized for that.

![ima![hp_results](https://hackmd.io/_uploads/HJLdrdteA.png)
ge](https://hackmd.io/_uploads/Bkwaho9yR.png)
Figure 1 (From: https://arxiv.org/abs/1708.06500v2)

#### Implementation
The Sparse Convolutional layer is nicely explained in Figure 1(b). The Sparse Convolutional layer in essense come down to a few steps:
1. Perform an elementwise multiplication between the input and the mask to remove the invalid pixels
2. Use a normal convolutional layer on this new input but without any built-in bias(as the bias should come later to avoid it also being normilized)
3. Normalize the convolutional layer by dividing by the amount of valid pixels in each window. The amount of valid pixels in each window is decided by using a convolutional layer (with all weights of 1). This convolutional layer has the same parameters as the convolutional layer that is used on the feature map. 
4. Add the bias to the new normalised feature map and you now have the new feature map the is the output.
5. To get the new mask, you can perform a max pool operation on the input mask. However in our implementation we simple clamp all values larger than 0 to 1 in the convolution of the mask that was used in the normalization step.

All these steps can be performed using existing modules. However, when we implemented the exact steps described above, normalization prevented the whole network from learning. It seemed like the normalization step reduced the weights in the convolutional layer too much (most weights had values around 1e-5). The only solution we were able to find was multiplying the convolutional layer by the kernel size to offset this weight reduction. It could be that this is caused by the convolutional layer that is used in our implementation already having a built-in normalization and by multiplying by the kernel size we undo this normalization. However, this shouldn't have been the case as the convolutional layer generally doesn't have any built-in normalization in the field of machine learning. 

(We originally also had a different implementation that avoided the use of existing convolutional layers and did it all thing using just NumPy. However, while that implementation should have allowed more fine-tuned control over the implementation we were not able to get it to run nicely on the GPU making it so slow that even testing it was infeasible. Therefore we moved to using existing convolutional layers as a basis.)



### Performance vs default CNN
#### Used Dataset
The dataset that is used to compare the two different CNNs is MNIST. MNIST is a dataset of handwritten numbers that is an often used dataset to test simple classification networks. The dataset consists of 28x28 black and white images and has a training set of 60,000 examples and a test set of 10,000 examples. 

The simple nature of the dataset allows us to train the model more quickly which allows for a faster feedback loop during the development of the sparcity CNN. This was desirable as we would be developing a CNN without a pre-existing template so bugs were expected. It also allowed us to put less focus on optimizing our implementation.

The MNIST dataset isn't very information-dense, however, which requires some consideration when testing the sparsity CNN against a default CNN. Below we added a visualization of a batch of MNIST samples at different levels of sparcity(50%, 70% and 90%). The samples at 50% still look relatively intact even though a large amount of the pixels were deleted since a large portion of those pixels didn't really add any information to the sample. It is really only around the 70% sparsity level that it adds to the difficulty and at 90% it starts to become almost impossible to classify some samples. But this is highlighted to support our decision to only compare the dataset on the sparsity levels of 70% to 90% and forgo testing on the lower sparsity levels.

sparcity of 50%
![Untitled](https://hackmd.io/_uploads/ByIumy91R.png)
![Untitled-1](https://hackmd.io/_uploads/H1nYXycJC.png)

Sparcity of 70% 
![Untitled-1](https://hackmd.io/_uploads/Byo7V1c10.png)

sparcity of 90%
![Untitled](https://hackmd.io/_uploads/BJbhQyqJ0.png)
![Untitled](https://hackmd.io/_uploads/H1q0Qy9kR.png)


**Results** 
The training and testing accuracy and the loss of the models are computed and evaluated. It was shown that the accuracies remained the same for a total of 55 epochs for the default CNN with the same number of layers as that of the Sparsity CNN. The sparsity CNN performed decent and the default CNN did not seem to learn anything and always hovered around the percentage that could be gotten from random guessing.

| Default CNN 70%     | Default CNN 90% |
| --- | -------- |
| 9.87  | 9.87  |

| Sparsity CNN 70%     | Sparsity CNN 90% |
| --- | -------- |
| ![Untitled-1](https://hackmd.io/_uploads/H1YUEoFJA.png) ![Untitled-1](https://hackmd.io/_uploads/rk8S4oYJR.png)   | ![Untitled-1](https://hackmd.io/_uploads/HJL5h5tyR.png)   ![Untitled](https://hackmd.io/_uploads/rJWj35tJR.png)  |

**70% Sparsity:** 
* The training loss decreases sharply initially and then plateaus, which is typical for neural network training. This indicates that the network is learning effectively from the less sparse (70%) data and is converging to a solution.
* The training accuracy improves rapidly and then continues to increase at a slower pace, almost plateauing towards the end. The test accuracy closely follows the training accuracy indicating that the network generalizes well to unseen data. There is a small but noticeable gap between training and test accuracy hinting at some level of overfitting, although this seems relatively minor.

**90% Sparsity:**
* Similarly, the training loss decreases sharply at first indicating learning. However, the overall loss values are higher than the 70% sparsity scenario suggesting that the network finds it more challenging to learn from the more sparse (90%) data.
* Both training and test accuracies increase as the epochs progress, but the rate of improvement slows down as it reaches higher epochs. The gap between training and test accuracy is wider compared to the 70% sparsity case suggesting that the network may be overfitting more to the training data or that the increased sparsity makes generalization more difficult.

Performance comparison(MAE) between different CNN methods.


| Sparsity levels at train and test: | 70% | 80% | 90% |
| ---------------------------------- | --- | --- | --- |
| ConvNet                            |  0.9983  |   0.9984  |   0.9983  |
| SparseConvNet                      |  0.0647   |  0.0648   |  0.065   |

**Observations:**
    
The ConvNet showing MAE values of 0.998 across all sparsity levels is exceptionally high, especially with the MAE being interpreted in the context of classification tasks where we are considering the deviation of the model's predicted probabilities from the true class labels (in a one-hot encoded form). This would typically suggest that the ConvNet is almost always incorrect, assigning very low probabilities to the true class.
However, considering the consistent value of 0.998 across different sparsity levels; it seems there might be a misunderstanding or misinterpretation of the results?
    
The SparseConvNet shows MAE values around 0.065 i.e. significantly lower than those of the ConvNet indicating much better performance. These values suggest that on average, the SparseConvNet assigns high probabilities to the correct class with a small average error from the ideal probability of 1 for the true class.
The slight increase in MAE with higher sparsity levels (from 0.0647 to 0.065) is expected, as more sparsity generally introduces more challenge to the model. However, the SparseConvNet appears robust to sparsity showing only a minor degradation in performance as sparsity increases.
    
<We NEED hyperparameter tuning most probably>

### Hyperparameter Tuning
Improving the sparse invariant CNN (convolutional neural network) model by tuning its hyperparameters. 

**Challenges**
The original CNN model was changed so the hyperparameters could easily be adjusted as arguments to a model class, the python notebook used for this can be found on the GitHub under 'hp_optimization.ipynb'. It was quickly realized that training the model, even on Google Colab’s cloud computing, took about 30 seconds for each epoch (a full pass through the training data), which makes computing this for a lot of epochs in total and a large combination of hyperparameters unsuitable.
Multiprocessing was used to save time, however it did not turn out to be efficient in this specific case. Therefore, it was limited how many different hyperparameters could be tested, and how many epochs could be ran. To start off, 15 epochs were used for each iteration to get a basic idea of how well the model was doing, then increasing this to 25 epochs to make sure the loss was properly converged.

**Choosing Hyperparameters**
The following hyperparameters are used to optimize the performance of the model: 
    - The number of layers in the network
    - The activation functions
    - The training batch sizes
    - The learning rate. 
The tuning started off by testing a wide domain on each parameter, with each iteration narrowing the search down. Note that the selection of hyperparameters is limited here, some sacrifices had to be made due to how computationally expensive the training process was, therefore it’s important to disclaim there are additional hyperparameters which can still be optimized later.

**Tuning process**

In the first steps a total of 4 (random) grid searches were performed on the hyperparameters stated above, the ranges that were selected were based of a combination of values which have previously worked well and some random values to explore new domains. For each iteration the best values for the hyperparameters were saved, the raw results are shown in log1.txt to log4.txt. During these 4 iterations, different combinations as well as more specific domains were explored, this way it was possible to find a local (or global) maximum around some set of parameter which had been proven to show promising results before.

**Further optimization**
After this initial search, Bayesian optimization was used to tune the hyperparameters further. This optimization tool is efficient because it uses the results of previous evaluations to make better guesses about the ‘guess’ for the next set of hyperparameters, it is known to be particularly useful when computational power may be an issue. The raw results of the Bayesian optimization can be found in log5.txt, log6.txt and log7.txt.

**Results**
By appending all raw together, the results were evaluated more thoroughly using visualization tools. A parallel coordinate plot was made to show how each hyperparameter interacts with each other and which domains of parameters were explored.
    

![hp_results](https://hackmd.io/_uploads/SJA1LutxA.png)

The figure above shows all possible combinations taken during the hyperparameter optimization, the aim is to minimize the (log) loss for best performance. From the figure can be observed that there are some sets of hyperparameters which perform significantly worse than others, for example the use of ELU as an activation function was found to be ineffective for this model and therefore taken out in earlier iterations of the tuning process. The overall best performing hyperparameter combinations are shown in the table below.

| Number of Layers | Activation Function | Batch Size | Learning Rate | Loss   |
|------------------|---------------------|------------|---------------|--------|
| 7                | LeakyReLU           | 74         | 0.00106       | 1.0605 |
| 7                | LeakyReLU           | 59         | 0.00100       | 1.0630 |
| 7                | LeakyReLU           | 90         | 0.00121       | 1.0647 |

This table shows that each one of the best performing hyperparameter sets uses 7 layers and a LeakyReLU activation function in the network architecture, the batch size and learning rate are shown to be in a local optimum when tuned to ~74 and ~0.001 respectively.
  
    
**Conclusion**
The main method of improving the sparse invariant CNN's performance was the use of hyperparameter tuning. The best performing configurations were identified using grid searches and Bayesian optimization, this proved the usefulness of the LeakyReLU activation function as well as particular learning rates, batch sizes, and the number of layers. Despite computing difficulties, this approach made significant progress in increasing the performance of the model on sparse data.

    

    





    
## New sparsity CNN vs default CNN on Caltech-101 dataset

In this section, we explore the assessment of our new sparsity CNN alongside the default CNN, utilizing the Caltech-101 dataset with a 70 percent sparsity. Through a thorough examination of their performance on this dataset, we aim to uncover how effectively these architectures handle the complexity and variability inherent in real-world image data. Additionally, we seek to determine whether the sparse CNN outperforms the default CNN on the sparsified images.
#### The Dataset
    
The Caltech-101 dataset is widely recognized in computer vision for its diverse collection of images tailored for object recognition tasks. Comprising 101 distinct object categories, each category encompasses approximately 40 to 800 images. These images exhibit considerable variation in scale, viewpoint, and lighting conditions, posing a formidable challenge for algorithms aiming to accurately classify objects.

![images_per_clas](https://hackmd.io/_uploads/rkZMWDKeC.png)

    
A visual representation of the class distribution reveals a marked imbalance, with certain classes significantly outnumbering others. This imbalance can lead models to favor classes with greater representation in the dataset, potentially compromising the model's overall performance.

To address this issue, a solution was implemented to mitigate the impact of class imbalance. By limiting the dataset to a maximum of 100 images per class, a more balanced distribution was achieved. While image augmentation could further enhance model performance, its implementation was deemed resource-intensive and beyond the scope of the project.    

#### Results
 
Upon completing the training and testing of both models, the obtained results are as follows:
    
| Sparsity CNN 70%     | Default CNN 70% |
| --- | -------- |
| ![trainingloss_spars_Cnn](https://hackmd.io/_uploads/BkwKQDYl0.png) ![accuracy_spars_cnn](https://hackmd.io/_uploads/HyfBQvFlR.png)   | ![trainingloss_normalcnn](https://hackmd.io/_uploads/B1-RXDFgC.png)   ![accuracy_normal_cnn](https://hackmd.io/_uploads/BJ1h7vFlA.png)  |
    

In conclusion, after 55 epochs of training, the sparsity CNN demonstrates a test accuracy of approximately 6.5% and a training accuracy of 5.5%, while the normal CNN exhibits roughly 3% accuracy for both test and training sets. 
This suggests that the sparse CNN may outperform the default CNN. However, the results from both CNNs indicate potential architectural issues, as the accuracy should ideally be higher after 55 epochs. 
    
To investigate further, a quick examination of the guessed classes for a batch revealed that the models predicted classes beyond the expected range. This discrepancy suggests that the last layer of the models might not be appropriately sized for a classification problem. In particular, the output size was calculated as the product of the image height, image width, and the number of classes. While this was not a significant concern with datasets like MNIST due to fewer classes and smaller images, with larger images (150 by 150), the output layer potentially ballooned to over 2 million possible classes instead of the expected 101. This oversight went unnoticed for some time, contributing to the observed issues. Addressing this misalignment between the model architecture and the classification task is essential for future improvements.
