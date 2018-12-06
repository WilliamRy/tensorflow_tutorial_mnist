# tensorflow_tutorial_mnist
This tutorial is made for the following usage:
1. an easy-setup test for the env of gpu, cuda, cudnn and tensorflow-gpu  
2. an example to show how to use some basic functions of tensorflow

### Setup
[`requirements.txt`](requirements.txt) to be added...

### Usage

*  The [`main.py`](main.py) script will load data, build model, train model, save model, clear default graph,
 load model and make predict on test data. Also it shows that:
    1. How to restrict the using of gpu memory
    2. How to record the training process so that you can monitor in tensorboard
    3. How to clear default graph to avoid the "variables exist" error
    4. How to partially restore the model by scope name
     
*  The [`model.py`](model.py) file contains the models, one can check the neural net structure in this file.

*  The [`feeder.py`](feeder.py) file defines a standard way to obtain data 