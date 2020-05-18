# **Traffic Sign Recognition** 

## Writeup / README

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/histogram.png
[image2]: ./writeup_images/gray.png
[image3]: ./writeup_images/lenet.png

[image4]: ./writeup_images/14.jpg "Traffic Sign 1"
[image5]: ./writeup_images/13.jpg "Traffic Sign 2"
[image6]: ./writeup_images/36.jpg "Traffic Sign 3"
[image7]: ./writeup_images/3.jpg "Traffic Sign 4"
[image8]: ./writeup_images/1.jpg "Traffic Sign 5"

[image9]: ./writeup_images/traffic.png
[image10]: ./writeup_images/softmax.png
[image11]: ./writeup_images/visual.png


### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a list of all the traffic signs in the data set and what the first instance of each instance of the labeled data looks like. There is also a bar chart to show the distribution of each image throughout the data set.

![alt text][image1]



### Design and Test a Model Architecture

#### 1. Preprocessing steps

As a first step, I decided to convert the images to grayscale because that would be one 2 less channels of data for the network to process. After inintially building the model to support a full color image, the conversion to grayscale was found to increase classification accuracy in the model by 3 percent alone. This supported my assumption that grayscale would produce a more accurate classification.  

As a last step, I normalized the image data because this would suppress effects of extremely bright images. If I did not normalize the data, I would potentially be training my network to detect bright images versus the objects in the image. One method for normaliziing the dataset was to subtract 128 from each pixel (range 0-255) and divide by 128, this simulates an approximate mean of zero and equal variance. In practice, it was found that the data could be trained more accurately by using the numpy mean and variance functions to set the mean to zero and variance to 1.

The difference between the original data set and the augmented data set is that the original data set has 3 color channels with a range from 0-255 and an arbitrary mean and variance, whereas the augmented data set has 1 color channel ranging from [-1, 1] with zero mean and an equal variance of 1. 

Here is an example of a traffic sign image after grayscaling and normalization.

![alt text][image2]



#### 2. Model Architecture

My final model was based on the LeNet-5 architecture and included dropout in the fully connected layers. 

The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Color images in the german traffic databases have 3 color channels, but the data was converted to grayscale, therefore there is 1 color channel. Dropout was only used on the fully connected layers as convolutional layers are not very dense and research has shown that dropout is only useful between fully connected layers.

My final model consisted of the following layers:

    * Layer 1: Convolutional. The output shape should be 28x28x6.

    * Activation. ReLu activation function.

    * Pooling. The output shape should be 14x14x6.

    * Layer 2: Convolutional. The output shape should be 10x10x16.

    * Activation. ReLu activation function.

    * Pooling. The output shape should be 5x5x16.

    * Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

    * Layer 3: Fully Connected. This should have 120 outputs.

    * Activation. ReLu activation function.

    * Dropout. Keep probability.

    * Layer 4: Fully Connected. This should have 84 outputs.

    * Activation. ReLu activation function.

    * Dropout. Keep probability.

    * Layer 5: Fully Connected (Logits). This should have 43 outputs.
 


#### 3. Training the model

To train the model, I used a learning rate of .001, 40 epochs, and a batch_size of 100. THe learning rate, epochs, and batch_size were heavily tuned during the training process to ensure an accurate model. It was found that by adding a 50% dropout in the fully connected layer, more epocs would increase the accuracy of the model by 2-3%. 

From the training model the Adam Optimizer was used, which is similar to stochastic gradient descent. Softmax cross entropy with logits was used as the loss function.  

My training pipeline looked like this:

    EPOCHS = 40
    BATCH_SIZE = 100
    rate = 0.001
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    
    
    
#### 4. Iterative Procedure

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 95.4$ 
* test set accuracy of 92.7%

The goal of the model was to have a validation set accuracy greater than 93%. To meet this goal an iterative process was chosen, where the first step was to implement a well known architecture on the data that would be well suited for the traffic sign problem. The Lenet architecutre was chosen since it was found to be extrememly effective against the MINST handwritten digit database. This database was visually similar enough to the traffic signs. The final model's accuracy on the training, validation, and test set provice substantial evidence that the model is working well on German traffic sign data.

![alt text][image3]

Problems that were ininitlaly faced with the architecutre included using color images - this was solved by preprocessing the data to be grayscale and normalizing the dataset. 

The traffic sign data is also more complicated then the handwritting data set, the adjust this appropriately, dropout layers were added between the fully connected layers. Dropout was only used on the fully connected layers as convolutional layers are not very dense and research has shown that dropout is only useful between fully connected layers. While the training set had nearly 100% accuracy, the validation set initially had 89% accuracy, this led me to believe that overfitting was occuring. Dropout helped mitigate the overfitting I found during training the dataset.

The batch size was lowered to decrease the number of images tuned during each iteration of the network. 

"The stochastic gradient descent method and its variants are algorithms of choice for many Deep Learning tasks. These methods operate in a small-batch regime wherein a fraction of the training data, usually 32--512 data points, is sampled to compute an approximation to the gradient. It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, as measured by its ability to generalize."
(From Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang. On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.)

Decreasing the batch size was found in my model to increase the quality and accuracy of my model when testing against the validation set. The number of epochs was also increased because after adding dropout to the model it was found that the model was still increasing in accuracy after 10 epochs. 

An important design choice in this iterative procedure was the addition of the dropout layers between the fully connected layers. The dropout layer keep probability was chosen to be 50% for the training data. This high dropout rate allows the model to create redundancies in the network of weights and biases which can be used to more accurately predict the images and make the general model more robust. 




### Test a Model on New Images

#### 1. Images found on the web

Here are five German traffic signs that I found on the web:

![alt text][image9]

Initially, I believed these images would all be easy to classify, but it turned out that my model had a difficult time classifying these images. These images may not be real German traffic signs, as I have never been to Germany to see their traffic signs and have no expertise in the subject. These images may also be inherintly difficult to classify since they all have watermarks from the websites were they were sourced.  

#### 2. Model Predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Yield     			| Ahead only    								|
| Go straight or right	| General caution   							|
| 60 km/h	      		| 60 km/h    					 				|
| 30 km/h   			| 30 km/h           							|



The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set of 92.7%, this leads me to believe I chose extremely difficult images or did not have enough images for an accurate metric. 



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is very sure that this is a stop sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| .00     				| General cuation   							|
| .00					| 20 km/h        								|
| .00	      			| Turn right ahead  			 				|
| .00				    | 70 km/h           							|

For the second image, the model is very sure that this is a ahead only sign (probability of .98), and the image does not contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Ahead only   									| 
| .01     				| General caution								|
| .00					| Stop sign  									|
| .00	      			| Children crossing				 				|
| .00				    | Turn left ahead    							|

For the third image, the model is very sure that this is a General caution (probability of 1), and the image does not contain a general cuation sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution								| 
| .00     				| 30 km/h 										|
| .00					| Stop sign										|
| .00	      			| Traffic signals				 				|
| .00				    | 20 km/h           							|

For the fourth image, the model is somewhat sure that this is a 60 km/h sign (probability of .39), and the image does contain a 60 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .39         			| 60 km/h   									| 
| .24     				| Right-of-way									|
| .19					| 30 km/h										|
| .16	      			| General caution				 				|
| .02				    | 80 km/h            							|

For the fifth image, the model is very sure that this is a 30 km/h sign (probability of 1), and the image does contain a 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 30 km/h    									| 
| .00     				| 70 km/h   									|
| .00					| 20 km/h										|
| .00	      			| 50 km/h   					 				|
| .00				    | General cuation      							|

The following image shows the grayscale normalized image along with a histogram of softmax probabilites:

![alt text][image10]




It can be seen that the images that were labeled incorrectly, the classifier was very sure that it was correct. This leads me to believe I could be seeing an issue with overfitting on the dataset or not enough generalized data. 




### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The test image being fed into the network was a 30 km/h sign, which can be seen below along with the feature map that was found for the first 2D convolution layer.  


![alt text][image11]


It can be seen from the feature map that the letters were used in this layer to make classifications along with the borders of the sign. This makes sense as these are the distinguishing parts of this traffic sign. 

