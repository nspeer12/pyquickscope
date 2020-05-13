# pyquickscope
Pyquickscope is a computer vision based quickscoping program designed to run on top of first person shooter games. A computer vision model trained on 10,000 images from Call of Duty Modern Warfare detects if another player is in a 300x300 pixel region around the crosshair. If the computer vision model is confident that another player is targetable, the program can execute a number of actions such as a programed quickscope sequence, a single shot, or a three round burst.

# Setup

## Windows
Note: This program is designed to run on Windows. It may work on Linux with minor modifications and on Linux compatible games. Some of the Python libraries included for controlling the mouse and taking screenshots are specific to Windows, so swapping with comparable libraries on Linux may be required.

## CUDA Libraries
Install **CUDA 10.1**. Other versions of CUDA may not be compativble with Tensorflow 2. I recommend following the link below and using the gui installer for Windows. Restart your PC when the installation completes.

CUDA 10.1: https://developer.nvidia.com/cuda-10.1-download-archive-base

## Anaconda
Anaconda is the recommended package manager for running this program. These steps may be modified to work with pip, or a mixture of pip and anaconda.

### Create Virtual Environment Anaconda

`conda create -y --name quickscope python==3.7`

### Install Python Packages to Virtual Environment

`conda install --force-reinstall -y --name quickscope -c conda-force --file requirements.txt`

#### Activate Virtual Enviornment

`conda activate quickscope`

# Model

To create the model for this program, I used the pretriained classification models included in `Tensorflow.keras.application` as convolutional base. More hidden layers of varying size were added after the convolutional base. Finally, a single ouput node with a sigmoid activation function is added. The single output represents our a logistic regression of the class prediction. An output of 1 is a high probability that a player is in the frame, while an output of 0 is a low probability. The entire model is then trained on the new data set. Since the convolutional base is already trained on ImageNet, training the model on the data collected in the video games is using the method of transfer learning. 

TODO: Create and train model from scratch.

#### Load in Model

Download any model from the following link and put it into a directory called *models*

https://drive.google.com/drive/folders/136m34OQsPDzP5-ijoFwja9F48Dfstv-5?usp=sharing


Specify the directory of the model in `main.py` in the constant `model_file`.


#### Training on Data Set

The data set can be accessed with the following link:

https://drive.google.com/drive/folders/1xZwJSrQ-7Ys-CRlwWFsheDIP3M4EjeOu?usp=sharing


Place the data set into a directory called *data* and specify its location in `model.py`.

Change layer shapes and other hyper-parameters in `model.py` and run. A new model will be created an automatically saved. After training, specify the new model in `main.py` and start playing!

### Create a Data Set

#### Data Collection
Two python programs called `data-collection.py` and `data-cleaning.py` will help you create your own data set. Some games that might work well include *Halo*, *CSGo*, *Team Fortress 2*, and any other first person shooter video game.


`data-collection.py` has a keyboard listener that waits for a specific keyboard or mouse input. When the input is triggered, a screenshot is taken and saved to *data/raw/<class>*. This is similar to photographing a target. To get photos of the class *other* which is simply the abscence of any player, set up an empty lobby and walk around taking photos. To get photos of players, have a friend join the lobby and carefully photograph them in different lighting conditions, maps and player skins. 


#### Data Cleaning
`data-cleaner.py` is a program to manually and automatically sort raw collected data into trainable directories. The function `human_in_the_loop_labeling()` will loop through all images in the *raw* directory and display them. You can then press a number of keys to save them to the *train* directory under a specific class.

**v**: label as player
**b**: label as other
**n**: delete image

Once you have a sufficiently large data set to create an accurate model, the model can be used to help filter data. Specify the model to be used at the top of the program and run `label_with_model()` to have the model filter all *raw* images for you. The model will send images to 10 different directories based on the probability it assigns to it. Then you can run `human_in_the_loop_labeling()` to manually check the model's work. The model should never be able to move images to the *train* directory on its own, because it would only solidify its inaccuracy. Carefully check each image to ensure the model is labelling accurately.


### TODO
- create a downloadable .exe file
- better command line arguments

