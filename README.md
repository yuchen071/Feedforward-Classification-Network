# Feedforward Classification Network from Scratch
## Instructions
Implement a feedforward neural network (FNN) model by yourself to classify the MNIST dataset. Deep learning packages such as Pytorch and Tensorflow are not allowed.

1. Design a FNN model architecture and perform the random initialization for model weights. Run backpropagation algorithm and use mini-batch SGD (stochastic gradient descent) to optimize the parameters

    (a). Plot the learning curves and the accuracy of classification versus the number of iterations until convergence for training data as well as test data.  
    (b). Repeat 1(a) with different batch sizes.  
    (c). Repeat 1(a) by performing zero initialization for the model weights.  

## Requirements
MNIST dataset source: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz 

The root folder should be structured as follows:
```
root
  ├─ main.py
  └─ mnist.npz
```
### Dependecies
* numpy  
* matplotlib  
* Pillow

## Train
Run the following code to train with default parameters:  
```
python main.py
```

### Arguments
Custom network configs and image evaluation are available. Format examples are provided in the `docs` folder.

Parser example:
```
python main.py \
> --config <config>.json \
> --imglist <imglist>.txt
```

If `<imglist>.txt` is provided, the program will generate an `output.txt` file containing the predicted classes. An example file is provided in the `docs` folder.  
Images for evaluation must be `28 x 28 x 1` in shape.