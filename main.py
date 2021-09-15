# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:52:46 2021

@author: Eric
"""

import json
import argparse
import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from PIL import Image

import time

#%% Default params
PATH_TRAIN = "train.npz"
PATH_TEST = "test.npz"

BATCH_SIZE = 200
EPOCHS = 20
LR = 1e-3

NN = [[1024,2048],[2048,512],[512,6]]
ACT = ["relu", "relu", "softmax"]
# ACT = ["leaky relu", "leaky relu", "softmax"]
# ACT = ["softplus", "softplus", "softmax"]
CRIT = "cross_entropy"

# WEIGHT = "xavier"     # recommended for sigmoid, tanh functions
WEIGHT = "he"   # recommended for relu functions
# WEIGHT = "rand"
# WEIGHT = "zero"

#%% functions
def check_layer(nn_list):
    for i in range(len(nn_list)-1):
        if nn_list[i][1] != nn_list[i+1][0]:
            raise Exception("Layer Dimensions Inconsistent!")

def check_act_dim(act, layer_list):
    if len(act) != len(layer_list):
        raise ValueError("Expected %d activation function layers, instead got %d" % (len(layer_list), len(act)))

def img_process(img_list):
    out = []
    for i in range(len(img_list)):
        # row major
        out.append(np.ravel(img_list[i]) / 255)
        
    return np.array(out)

def xavier_weight(layer2, layer1):
    # xavier uniform weight initialization
    lower, upper = -np.sqrt(6.0/(layer1 + layer2)), np.sqrt(6.0/(layer1 + layer2))
    out = rand(layer2, layer1)*(upper - lower) + lower
    return out

def check_weight_dim(weight, layer_list):
    if len(weight) == len(layer_list):
        for l_id, data in enumerate(layer_list):
            if weight[l_id].shape != (data[1], data[0]):
                raise ValueError("Weight dimentions at layer %d of %d is wrong!" % (l_id+1, len(layer_list)))
    else:
        raise ValueError("Expected %d weight layers, instead got %d" % (len(layer_list), len(weight)))

def check_label_dim(label_list, nn_list):
    label_unique = np.unique(label_list)
    out_dim = nn_list[-1][-1]
    
    if len(label_unique)-1 != label_unique[-1]:
        print("Warning: Missing categories in dataset")
        
    if len(label_unique) < out_dim:
        print("Warning: Output dimension larger than dataset category amount")
        
    elif len(label_unique) > out_dim:
        raise Exception("Output dimension smaller than dataset category amount")
        
    pass

def init_wb(nn_list, w_input):   
    weight = []
    if isinstance(w_input, str):
        if w_input == "xavier":
            # Xavier uniform
            for _, dims in enumerate(nn_list):
                w = xavier_weight(dims[1], dims[0])
                weight.append(w)
                
        elif w_input == "he":
            # Xavier normal (He)
            for _, dims in enumerate(nn_list):
                std = np.sqrt(2.0 / (dims[0] + dims[1]))
                w = randn(dims[1], dims[0]) * std
                weight.append(w)
        
        elif w_input == "rand":
            for _, dims in enumerate(nn_list):
                # -1 ~ +1 uniform random
                w = rand(dims[1], dims[0])*2 - 1
                weight.append(w)
        
        elif w_input == "zero":
            for _, dims in enumerate(nn_list):
                # zero weights
                w = np.zeros((dims[1], dims[0]))
                weight.append(w)
        
        else:
            raise Exception("Weight initializer undefined: %s" % train_config["weight"])
    
    elif isinstance(w_input, list):
        weight = w_input
        
    else:
        raise Exception("Weight initializer undefined: %s" % str(train_config["weight"]))
        
    bias = []
    for _, dims in enumerate(nn_list):
        # init as zeros
        b = np.zeros((dims[1], 1))
        bias.append(b)
        
    return weight, bias
    
#%% dataloader
def dataloader(images, labels, batch_size):
    num_batch = np.ceil(len(labels) / batch_size).astype(np.int16)
    
    out = []
    for i in range(num_batch):       
        tmp = [
            images[batch_size*i : batch_size*(i+1), :].T,
            labels[batch_size*i : batch_size*(i+1)]
            ]
        out.append(tmp)
    
    return out

#%% neural network functions
def wab(w, a, b):
    return w.dot(a)+b

def label2target(label, nn_shape):
    # label vector to target matrix
    target = np.zeros(nn_shape)
    for i in range(nn_shape[1]):
        target[label[i]][i] = 1
        
    return target

def act_func(x, act):
    if act == "relu":
        y = np.maximum(0, x)
        
    elif act == "sigmoid":
        y = 1 / (1 + np.exp(-x))
        
    elif act == "tanh":
        e1 = np.exp(x)
        e2 = np.exp(-x)
        y = (e1 - e2)/(e1 + e2)
        
    elif act == "softplus":
        thresh = 20
        y1 = np.log(1 + np.exp((x <= thresh)*x))
        y2 = (x > thresh)*x
        y = y1 + y2
        
    elif act == "leaky relu":
        y1 = (x>0) * x
        y2 = (x<=0) * x * 0.01
        y = y1 + y2        
    
    elif act == "softmax":
        # overflow prevention
        # exp(a_i - max(a)) / sum(exp(a_i - max(a)))
        m = x.max(axis=0)
        y = np.exp(x - m)
        y = y/np.sum(y, axis=0)
    
    else:
        raise Exception("Undefined activation function: %s" % act)
    
    return y

def loss_func(x, target, crit):
    if crit == "cross_entropy":
        epsilon = 1e-15     # prevent log(0)
        # epsilon = np.finfo(float).eps
        y = -np.sum(target * np.log(x + epsilon)) / target.shape[1]
        
    elif crit == "MSE":
        y = np.mean(np.sum((x - target)**2, axis=0))
        
    else:
        raise Exception("Undefined criterion: %s" % crit) 
        
    return y

# da_dz
def de_act_func(x, act):
    if act == "relu":
        y = (x>0) * 1
        
    elif act == "sigmoid":
        s_x = act_func(x, "sigmoid")
        y = s_x * (1 - s_x)
        
    elif act == "tanh":
        y = 1 - act_func(x, "tanh")**2
        
    elif act == "softplus":
        y = act_func(x, "sigmoid")
        
    elif act == "leaky relu":
        y1 = (x>0) * 1
        y2 = (x<=0) * 0.01
        y = y1 + y2
        
    else:
        raise Exception("Act function not programmed yet: %s" % act)
        
    return y

# dE_da
def de_loss_func(x, target, crit):    
    if crit == "MSE":
        y = 2*(x - target)
        
    elif crit == "cross_entropy":
        # The gradient of the cross-entropy loss with 
        # respect to the *input* to the softmax function.
        # dE_dz
        y = x - target
        
    else:
        raise Exception("Loss function not programmed yet: %s" % crit)
    
    return y

#%% train
def train(train_config, dataset):
    # initialize train and test dataset
    trainloader = dataloader(dataset["train_images"], dataset["train_labels"], 
                             train_config["batch_size"])
    testloader = dataloader(dataset["test_images"], dataset["test_labels"], 
                             train_config["batch_size"])
    
    num_layers = len(train_config['nn'])    # input layer not included
    act_list = train_config["act"]
    lr = train_config["lr"]
    
    # initialize weights and biases
    weight, bias = init_wb(train_config['nn'], train_config["weight"])
    check_weight_dim(weight, train_config["nn"])
    
    # loss history
    train_ave_loss = []   # per epoch
    train_acc = []    # per epoch
    test_ave_loss = []
    test_acc = []
    
    for epoch in range(train_config["epoch"]):
        print("\nEpoch %3d: " % (epoch+1), end = "")
        
        train_loss = [] # per batch
        train_correct = 0  # count
        test_loss = []
        test_correct = 0
        
        # train
        for batch_id, data in enumerate(trainloader):
            x, label = data[0], data[1]
            batch_len = len(label)
            
            a = [] # include input layer
            z = [] # input layer not included
            
            # forward
            a.append(x)
            for layer_id in range(num_layers):
                z.append(wab(weight[layer_id], a[layer_id], bias[layer_id]))
                a.append(act_func(z[-1], act_list[layer_id]))
            
            # label vector to target matrix
            target = label2target(label, a[-1].shape)
            
            # loss & accuracy
            loss = loss_func(a[-1], target, train_config["criterion"])
            ans = np.argmax(a[-1], axis=0)
            train_loss.append(loss)
            train_correct += np.sum(ans == label)
            
            # backward
            # cross entropy has to be paired with softmax to backprop
            dE_dz = []
            if train_config["criterion"] == "cross_entropy":
                if act_list[-1] == "softmax":
                    dE_dz.append(de_loss_func(a[-1], target, "cross_entropy"))
                else:
                    raise Exception("Last layer before Cross-Entropy has to be Softmax")
            else:
                dE_da = de_loss_func(a[-1], target, train_config["criterion"])
                dE_dz.append(de_act_func(z[-1], act_list[-1]) * dE_da)
                    
            # get rest of dE_dz list
            for layer_id in reversed(range(num_layers-1)):
                dE_da = weight[layer_id+1].T.dot(dE_dz[-1])
                dE_dz.append(de_act_func(z[layer_id], act_list[layer_id]) * dE_da)
            dE_dz = dE_dz[::-1] # reverse list
            
            # get dE_dw, dE_db list
            dE_dw = []
            dE_db = []
            for layer_id in range(num_layers):
                dE_dw.append(dE_dz[layer_id].dot(a[layer_id].T))
                dE_db.append(np.sum(dE_dz[layer_id], 1, keepdims=True))
                
                # mini-batch SGD
                weight[layer_id] = weight[layer_id] - lr * dE_dw[layer_id] / batch_len
                # bias[layer_id] = bias[layer_id] - lr * dE_db[layer_id] / batch_len
        
        ave_loss = np.mean(train_loss)
        acc = train_correct / len(dataset["train_images"])
        print("Train loss: %.4f" % ave_loss, end = ", ")
        print("Train acc: %.2f%%" % (acc*100), end = ", ")
        
        train_ave_loss.append(ave_loss)
        train_acc.append(acc)
        
        # test
        for batch_id, data in enumerate(testloader):
            x, label = data[0], data[1]
            batch_len = len(label)
            
            a = [] # include input layer
            z = [] # input layer not included
            
            # forward
            a.append(x)
            for layer_id in range(num_layers):
                z.append(wab(weight[layer_id], a[layer_id], bias[layer_id]))
                a.append(act_func(z[-1], act_list[layer_id]))
            
            # label vector to target matrix
            target = label2target(label, a[-1].shape)
            
            # loss & accuracy
            loss = loss_func(a[-1], target, train_config["criterion"])
            ans = np.argmax(a[-1], axis=0)
            test_loss.append(loss)
            test_correct += np.sum(ans == label)
            
        ave_loss = np.mean(test_loss)
        acc = test_correct / len(dataset["test_images"])
        print("Test loss: %.4f" % ave_loss, end = ", ")
        print("Test acc: %.2f%%" % (acc*100), end = "")
        
        test_ave_loss.append(ave_loss)
        test_acc.append(acc)
    
    # plot
    plt.figure()
    plt.plot(train_ave_loss, label="Training")
    plt.plot(test_ave_loss, label="Testing")
    plt.title("Average Loss History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()
    
    plt.figure()
    plt.plot(train_acc, label="Training")
    plt.plot(test_acc, label="Testing")
    plt.title("Average Accuracy History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()
    
    
    # if imgfilelistname
    if len(dataset["eval_images"]) != 0:
        ans = []
        for img_id, img in enumerate(dataset["eval_images"]):
            
            a = [] # include input layer
            z = [] # input layer not included
            
            a.append(img.reshape(len(img),1))
            for layer_id in range(num_layers):
                z.append(wab(weight[layer_id], a[layer_id], bias[layer_id]))
                a.append(act_func(z[-1], act_list[layer_id]))
            
            ans.append(np.argmax(a[-1]))
        
        with open("output.txt", "w") as fp:
            text = str()
            for i in ans:
                text += str(i)
                fp.write(str(i))
        
        print("\n\nEval Images Result: %s" % text)

#%% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--imgfilelistname", type=str)
    args = parser.parse_args()
    
    # config file
    train_config = dict()
    if args.config:
        print("Config file read from: %s" % args.config)
        with open(args.config, "r") as fp:
            train_config = json.load(fp)
        
        nn_list = []
        af_list = []
        for layer in train_config["nn"]:
            af_list.append(train_config["nn"][layer]["act"])
            nn_list.append([train_config["nn"][layer]["input_dim"],
                            train_config["nn"][layer]["output_dim"]])
        
        train_config["act"] = af_list
        train_config["nn"] = nn_list

    else:
        print("Default config")
        train_config["batch_size"] = BATCH_SIZE
        train_config["criterion"] = CRIT
        train_config["epoch"] = EPOCHS
        train_config["lr"] = LR
        train_config["nn"] = NN
        train_config["act"] = ACT
    
    check_layer(train_config["nn"])
    check_act_dim(train_config["act"], train_config["nn"])
        
    # weight file
    if args.weight:
        print("Weight file read from: %s" % args.weight)
        
        weight_data = np.load(args.weight)
        weight_list = []
        for item in weight_data.files:
            weight_list.append(weight_data[item].T)
            
        train_config["weight"] = weight_list
    
    else:
        print("Default weights initializer: %s" % str(WEIGHT))
        train_config["weight"] = WEIGHT
        
    # read train and test npz
    dataset = dict()
    train_data = np.load(PATH_TRAIN)
    test_data = np.load(PATH_TEST)
    
    dataset["train_images"] = img_process(train_data['image'])
    dataset["train_labels"] = train_data['label']
    dataset["test_images"] = img_process(test_data['image'])
    dataset["test_labels"] = test_data['label']
    
    check_label_dim(dataset["train_labels"], train_config["nn"])
    check_label_dim(dataset["test_labels"], train_config["nn"])
    
    # img list file
    if args.imgfilelistname:
        print("Image list read from: %s" % args.imgfilelistname)
        img_list = []
        with open(args.imgfilelistname, "r") as fp:
            img_name_list = [line.rstrip('\n') for line in fp.readlines()]
        
        for filename in img_name_list:
            img = Image.open(filename).convert("L")
            img_list.append(np.array(img))
        
        dataset["eval_images"] = img_process(img_list)
    
    else:
        dataset["eval_images"] = []
            
    # model train mode
    start_time = time.time()
    train(train_config, dataset)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"\n\nElapsed time: {elapsed}")