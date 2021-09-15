# Feedforward Classification Network
## 2021 Spring, Deep Learning Homework 1
### HW Instructions
You are given the dataset of medical images (MedMNIST.zip). This dataset contains 6 classes. In this exercise, you need to implement a feedforward neural network (FNN) model by yourself to recognize radiological images, and use specified algorithms to update the parameters. Please use train.npz as training data and test.npz as test data.

![MedMNIST image](https://github.com/yuchen071/Feedforward-Classification-Network/blob/main/docs/imgs/MedMNIST.png)

1. Design a FNN model architecture and perform the random initialization for model weights. Run backpropagation algorithm and use mini-batch SGD (stochastic gradient descent) to optimize the parameters

    (a). Plot the learning curves of J(w) and the accuracy of classification versus the number of iterations until convergence for training data as well as test data.  
    (b). Repeat 1(a) by using different batch sizes.  
    (c). Repeat 1(a) by performing zero initialization for the model weights.  

2. Implement a flexible program that can parse the arguments to generate a specific FNN model but without bias for each neuron, and also need to run backpropagation algorithm and use mini-batch SGD to optimize the parameters

## Requirements
This project requires a custom dataset `train.npz` and `test.npz` compiled from the MedMNIST dataset, which is not available in this github page. However, the MNIST handwritten digits dataset should also work.

The root folder should be structured as follows:
```
root
  ├─ main.py
  ├─ train.npz
  └─ test.npz
```

`train.npz` and `test.npz` should contain `image` and `label` columns:
```
train.npz
  ├─ image
  |   ├─ img 1
  |   ├─ img 2
  |   └─ ...
  └─ label
      ├─ label 1
      ├─ label 2
      └─ ...
```

### Dependecies
* numpy  
* matplotlib  
* Pillow

## Train

```
python main.py
```

### Arguments
Custom network configs, weights, and image evaluation are available. Format examples are provided in the Appendix folder.

Parser example:
```
python main.py \
> --config <config>.json \
> --weight <weight>.npz \
> --imgfilelistname <imgfilelistname>.txt
```

If `imgfilelistname.txt` is provided, the program will generate an `output.txt` file containing the predicted classes. An example file is provided in the Appendix folder.