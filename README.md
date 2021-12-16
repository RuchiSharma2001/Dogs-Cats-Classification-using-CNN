# Dogs-Cats-Classification-using-CNN
## Step 1:
### Data Preprocessing
In ths step dataset is loaded accordng to different directories and categories. After the data is read it is converted into array and stored in the list. Size of each image is set equal. Size of the image is as small as the image can be verified distinctly. Code for loading and storing data and getting ready for the training is completed.
## ABSTRACT:
The goal of this project is to build a model to classify whether the given image is of cat or of dog using Convolution Neural Network (CNN) .
Convolutional Neural Network is one of the main categories to do image classification and image recognition in neural networks. Scene labeling, objects detections, and face recognition, etc., are some of the areas where convolutional neural networks are widely used.
In CNN, each input image will pass through a sequence of convolution layers along with pooling, fully connected layers, filters (Also known as kernels). After that, we will apply the Soft-max function to classify an object with probabilistic values 0 and 1.

## INTRODUCTION:
It is basic CNN project which classifies between dog and cat. To make this project jupyter lab is used. Code is done in python language with the help of various libraries like open cv, matpotlib, tensorflow, os, numpy. In this we have use various steps such as data preprocessing, training the data, convert the convoluted image to feature map using input image and feature detector. After that maximum pooling will be done to convert image into smaller size. After max pooling flattening will be done to convert 2D matrix to 1D array. After flattening image s passed to dense layers and the value return using sigmod activation type are either 0 or 1. 0 is for dogs and 1 for cat.  
## PROBLEM DESCRIPTION: 
In this project we have to make a model using Convolution Neural Network that will check whether the entered image is of dog or cat. In this model we have to use various libraries and have to make a model. In this model we have generated the value of p for each image. If the value of p is zero then the image is of god else if p is 1 it is of cat.
## MOTIVATION:
New challenges and new things to learn are my favourite things to do . CNN is the vast topic I wanted to learn it. So I decided to start with the basic project work. Because implementing the things you have learned is the way to learn a subject for lifetime. While doing this project, I got to learn about new things like how to work on image, how to process and use image dataset.  I have learned how to work on tensorflow .

## SOFTWARE REQUIREMENTS:
•	Jupyter Notebook
•	Libraries (NumPy, Matplotlib, Tensorflow, Keras, os, cv2, random)
## HARDWARE REQUIREMENTS:
•	2 GHz Intel or high processor
•	Minimum of 180 GB HDD
•	At least should have 2 GB RAM
## LANGUAGE USED:
•	Python

## LIBRARIES USED:.
### Numpy – 
NumPy is a Python Programming Language library that is used to provide us a simple yet powerful data structure and is also used to perform a number of mathematical operations on arrays.
### Matplotlib. pyplot – 
It is a plotting library used in Python Programming Language and it it is used to provide an object oriented apifor displaying bar plots.
### Tensorflow-- 
Tensorflow is an open-source library for numerical computation and large-scale machine learning that ease Google Brain TensorFlow, the process of acquiring data, training models, serving predictions, and refining future results. Tensorflow bundles together Machine Learning and Deep Learning models and algorithms.
### Random-- 
For integers, there is uniform selection from a range. For sequences, there is uniform selection of a random element, a function to generate a random permutation of a list in-place, and a function for random sampling without replacement.
### OS-- 
The OS module in Python provides functions for interacting with the operating system. OS comes under Python’s standard utility modules. This module provides a portable way of using operating system-dependent functionality. The *os* and *os.path* modules include many functions to interact with the file system.
### Open CV-- 
OpenCV is a huge open-source library for computer vision, machine learning, and image processing. OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. It can process images and videos to identify objects, faces, or even the handwriting of a human.
 
 
## DATASET USED:
I have used dataset which have 2000 images of dogs and cats to predict the data[1]. Dataset used has two categories dogs and cats.
 
## APPROACH :
1.	Data gathering
2.	Data pre-processing
3.	Training data
4.	Convolution
5.	Max pooling 
6.	Flattening
7.	Send to dense neural layers
8.	Compile and fit generate the model
9.	Final code in which image will be predicted

## METHODOLOGY :
### 1.	Data gathering:- 
Data is collected from the online source which contain 2000 images each of dog and cat.

### 2.	Data pre-processing:- 
In this data of the images is read and the data is converted into array list for further execution. In this we will iterate over each and every image and resize to the smaller size which will make training easy in this case we have used 100. The shape of image 100, 100, 3. Because we have set its height and breadth to 100 and there are three channels for each image red, green blue.

### 3.	Training data:- 
In training data each image is appended and stored in the array test_data[]. We will iterate over each image in the data set and will store each image into this array to train the data.

### 4.	Convert convoluted images into featured map:-  
Every image will be converted into feature map using feature detector. Feature detector are always of size 3x3, 5x5, 7x7 or so on. Convolution is a linear operation involving the multiplication of weights with the input. The multiplication is performed between an array of input data and a 2D array of weights known as filter or kernel.  Using this we will generate the feature map which will consist of all the characteristic of the image if the value is zero the image is of dog or else the image is of cat.
 
### 5.	Max pooling:-  
In max pooling convoluted image is converted into smaller size using the matrix of any size like in this case we have taken 2x2. Pooling basically helps reduce the number of parameters and computations present in the network. It progressively reduces the spatial size of the network and thus controls overfitting. There are two types of operations in this layer; Average pooling and Maximum pooling. Here, we are using max-pooling which according to its name will only take out the maximum from a pool. 

### 6.	Flattening:- 
Flattening is merging all visible layers into the background layer to reduce file size. In this 2D feature map is converted in to 1D array for easy processing.
 
### 7.	Send to dense neural layers:-  
After all the processes image will be passed to hidden  the hidden layers for further data processing after which value of the  p will be generated. If the value of p if 0 image is of cat or else if it is of 1 it is of dog. After this model will be compiled and fit generated. Model then will be saved for further execution.
 
### 8.	Compile and fit generate model:-  
Model will then be compile fit generated to check accuracy of our code and then will be saved. This model have accuracy of  95.25% which is good enough. 
 

### 9.	Final code in which image will be predicted:- 
After the model is made we have created another ipynb file which model will be loaded and the path of the image that need to predicted is given and the value of p is calculated image. If the value of p if 0 image is of cat or else if it is of 1 it is of dog. After this model will be compiled and fit generated.

## CONCLUSION:
The model for classification between cats and dogs is made using CNN which can be used to detect any image. In the model we have used different libraries such as matpotlib, opencv and many more. This project helped in gathering many new information and helped in learning and building a successful model for image classification.
## REFERENCES:
1.	https://youtu.be/FLf5qmSOkwU
2.	https://www.javatpoint.com/pytorch-convolutional-neural-network
3.	https://www.analyticsvidhya.com/blog/2021/06/beginner-friendly-project-cat-and-dog-classification-using-cnn/
