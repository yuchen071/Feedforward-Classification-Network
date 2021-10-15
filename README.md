# Feedforward Classification Network from Scratch
## About the Project
Implement a feedforward neural network (FNN) model by yourself to classify the MNIST dataset without using deep learning packages such as PyTorch or Tensorflow.

## Requirements
MNIST dataset source: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz 

The root folder should be structured as follows:
```
📁 root
  ├─ 📄 main.py
  └─ 📄 mnist.npz
```
### Dependecies
```
numpy==1.20.2
matplotlib==3.3.4
Pillow==8.3.2
```

## Usage
### Train
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

## Results
[Report](https://github.com/yuchen071/Feedforward-Classification-Network/tree/main/results)

## Resources
The Neural Network video playlist from 3blue1brown helped me greatly in completing this project and understanding the fundamentals of backpropagation. Highly recommended if you want to give this project a try yourself.

[3blue1brown Youtube channel](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)  
[Neural Network Playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
