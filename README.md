## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project uses CNN to classify traffic signs using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report  

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Demo
This demo script classify a cropped image using trained model:
```bash
> python classify_traffic_sign.py -h
usage: classify_traffic_sign.py [-h] [--filename FILENAME]

optional arguments:
  -h, --help           show this help message and exit
  --filename FILENAME  cropped traffic sign image
```

### Data Set Summary & Exploration
Summary statistics of the traffic signs data set:
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43  
  
Some images and distribution of dataset:
![plot_sample_images](/examples/plot_sample_images.png)
![distribution](/examples/distribution.png)
The graph shows that distribution of traffic signs are similiar between training, validation and testing set.
And 5 most or least common traffic signs in training data are:   

**Most common five:**
 - 5.78%, 2010 images --- Speed limit (50km/h)
 - 5.69%, 1980 images --- Speed limit (30km/h)
 - 5.52%, 1920 images --- Yield
 - 5.43%, 1890 images --- Priority road
 - 5.34%, 1860 images --- Keep right

**Least common five:**
* 0.60%, 210 images --- End of no passing
* 0.60%, 210 images --- End of no passing by vehicles over 3.5 metric tons
* 0.52%, 180 images --- Speed limit (20km/h)
* 0.52%, 180 images --- Dangerous curve to the left
* 0.52%, 180 images --- Go straight or left

### Data Preprocessing
#####  Apply grayscale   
Using grayscale will lose information about the color but make it easier for the network to learn. 
The information carried in grayscale image is enough for the classification task since 
it can still get high accuracy with grayscale. After a couple of trials, 
I decide to apply grayscale since it's easier to learn and computationally cheaper.  
![grayscale](/examples/grayscale.jpg)
  
##### Apply Normalization  
Normalization helps the nerwork converge, here I normalize the pixel value roughly between -1 and 1.

### Model Architecture
The model is based with on LeNet with following changes:  

##### change the size of fully-connected layer to 1024
Since the num of classes increases from 10 to 43 so the original sizes of 120, 80 may not have enough 
learning capicity, so I increased both sizes to 1024.  

##### change the weight initial method to xavier
This change targets at gradient explode/vanish problem. It's introduced in this [paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 
and reduce the problem by making input variance same as output variance of each layer. And it's
the default weight initializer of `tf.layers.conv2d`.

#### apply batch normalization to convolution and fully-connected layers
This one speeds up learning and also acts as a regularizer. It's introduced in this [paper](https://arxiv.org/pdf/1502.03167v3.pdf).
The technique consists of adding an operation in the model 
just before the activation function of each layer, simply zero-centering and normalizing the inputs, 
then scaling and shifting the result using two new parameters per layer (one for scaling, the other for shifting)
It makes each layer learn from previous layer 
with a relatively fixed distribution than a moving target. 

#### change activation function to elu for fully-connected layers
`elu` is introduced in this [paper](https://arxiv.org/pdf/1511.07289v5.pdf) and author show that 
it outperform all ReLU variants in their experiments. After some trials, I find it doesn't work very well
with on conv layer. But when applied to only fc layer it provides a higher validation accuracy so I
only apply it to fc layers.

#### apply dropout to fully-connected layers except for the last one
#### apply l2 regularization to convolution and fully-connected layers
The last two was applied after I tried the model some times and find that the model is overfitting.
Dropout randomly ignore some nodes in a layer. It makes network can't rely on a specific input and become like
an ensemble of different sub-networks. Regularization makes weights smaller and both make the network 
generalize more. After trials, I end up with a 0.5 dropout rate and 1e-5 regularization param.  

| Layer         		    |     Description	        					| 
|:---------------------:|:--------------------------------------------:| 
| Input         		    | 32x32x1 grayscale image   							     | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	 |
| Batch Normalization		|												|
| RELU		 	 		        |												|
| Max pooling	      	  | 2x2 kernel size, 2x2 stride, outputs 14x14x6 |
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 10x10x6   |
| Batch Normalization		|												|
| RELU		 	 		        |												|
| Max pooling	      	  | 2x2 kernel size, 2x2 stride, outputs 5x5x16|
|	Flattern					    |	inputs 5x5x16, outputs 400								 |
| Fully connected		    | inputs 400, outputs 1024       						 |
| Batch Normalization		|												|
| ELU		 	 		          |												|
| Fully connected		    | inputs 1024, outputs 1024      						 |
| Batch Normalization		|												|
| ELU		 	 		          |												|
| Fully connected		    | inputs 1024, outputs 43      						   |

### Train and Test Model
Here I choose Adam optimizer since it works well. It combines monmentum and rmsprop and becomes a common
choice know. I train the model with 1e-3 learning rate, 50 epochs and 64 batch size. I compute validation 
accuracy every 10 steps, compare and save the best model so far. And when validation accuracy is higher
than 93% the batch size is set to 512. The final model results were:  
* training set accuracy of 99.55% 
* validation set accuracy of 97.37%
* testing set accuracy of 95.94%
![tensorboard_scalar](/examples/tensorboard_scalar.png)

### Test Model on New Images
Here I choose 5 images from testing data to visualize the softmax prediction, the images are:
![plot_test_images](/examples/plot_test_images.png)

And top 5 from softmax predictions for these images are:
![softmax.png](/examples/softmax.png)

The third traffic sign is partially blocked by the tree and it indeed gets a lower softmax
score. Characteristics of image that make it difficult to classify could be:
* partially blocked by other objects
* rotation and angle  
* low contrast

Since they are not met in training & validation data a lot so the model hasn't learned a lot
to deal with these cases.
