# Feedforward Classification Network
## 2021 Spring, Deep Learning Homework 1
### HW Instructions
You are given the dataset of medical images (MedMNIST.zip). This dataset contains 6 classes. In this exercise, you need to implement a feedforward neural network (FNN) model by yourself to recognize radiological images, and use specified algorithms to update the parameters. 

![MedMNIST image](https://github.com/yuchen071/Feedforward-Classification-Network/blob/main/docs/imgs/MedMNIST.png)

1. Design a FNN model architecture and perform the random initialization for model weights. Run backpropagation algorithm and use mini-batch SGD (stochastic gradient descent) to optimize the parameters

    (a). Plot the learning curves of J(w) and the accuracy of classification versus the number of iterations until convergence for training data as well as test data.  
    (b). Repeat 1(a) by using different batch sizes.  
    (c). Repeat 1(a) by performing zero initialization for the model weights.  

2. Implement a flexible program that can parse the arguments to generate a specific FNN model but without bias for each neuron, and also need to run backpropagation algorithm and use mini-batch SGD to optimize the parameters
