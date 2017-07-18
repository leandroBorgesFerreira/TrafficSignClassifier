
#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
 1.  Load the data set
 2.  Explore, summarize and visualize the data set
 3.  Design, train and test a model architecture
 4.  Use the model to make predictions on new images
 5.  Analyze the softmax probabilities of the new images
 6.  Summarize the results with a written report

---

###Data Set Summary & Exploration
I used numpy and Python to find some important information about the data set that was provided to execute the exercise. I got the following result:

Number of training examples = **34799**
Number of validation examples = **4410**
Number of testing examples = **12630**
Image data shape = **(32, 32, 3)**
Number of classes = **43**

You can take a look at 10 samples of the images of the exercise:
![enter image description here](https://lh3.googleusercontent.com/-ASINMj_OXcI/WW0BPbbevqI/AAAAAAAAADg/k6gz-A-sVE8COrH7ATqmGTTAJ5_W1XgFQCLcBGAs/s0/Screen+Shot+2017-07-17+at+15.24.34.png "Screen Shot 2017-07-17 at 15.24.34.png")

Some signs of the train data are more common than others. The following chart shows how the train data is distributed.

![enter image description here](https://lh3.googleusercontent.com/--z7DqJoSd1Y/WW0CnoQ4KUI/AAAAAAAAADs/YpkO_-02i-sIiz9kN71_lKsNXGnvDsMgwCLcBGAs/s0/Screen+Shot+2017-07-17+at+15.30.32.png "Screen Shot 2017-07-17 at 15.30.32.png")

It is reasonable to think that signs with a higher frequency will have a more accurate classification. 

###Design and Test a Model Architecture

At first, I decided to apply gray scaling in the imagens, but the precision of my training model dropped instead of rising. The idea was to use only one channel of color in the hope to make a simpler computation for my deep network, but it seams that the color helps the network to classify the data. 

So I decided to normalize the data. I used a very simple approach:

    def normalize(image_array):
        (image_array - 128.0) / 128.0
    
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    X_test = normalize(X_test)

## 2. Final model architecture
My final model consisted of the following layers:\
batch size: 128
epochs: 70

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected		| Input = 400. Output = 120.        			|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84.        			    |
| RELU					|												|
| Fully connected		| Input = 84. Output = 43.        			    |
|						|												|
 
 This is pretty much the LeNet architecture, a very powerful one. It didn't need a very complicated tunning, all I had to do was to choose a big number of epochs. 


## 3. Training

It took a long time to train my network. About 30 minutes in my computer. The precision takes a while to get big enough (bigger than 0.93), but it keeps increasing as the number of epochs also increases. At the 70th epoch, the precision was still increasing, but very slowly in the training set.

I got an accuracy of 0.937 for the test, 0.940 for validation and 0.940 for the training. 

This architecture is very well suited for this problem. With very little modification, the problem was solved with LeNet. It is a good choice for this problem as it scales well to big amount of data and gives a good precision.   
 

## 4. Test on New Images
Here are six German traffic signs that I found on the web:
![enter image description here](https://lh3.googleusercontent.com/-sRRbDIwRVd0/WW0jSEq-YvI/AAAAAAAAAEA/_A6FS1A2DJgIPvfYj9zEHyujXzDHneKpQCLcBGAs/s0/Screen+Shot+2017-07-17+at+17.50.17.png "Screen Shot 2017-07-17 at 17.50.17.png")

They are already resized to 32x32 pixels. All of the images were correctly predicted. I believe that the classifier presented 100% of accuracy because the images were very clean and easy to classify. To make the classification easier to the neural network, I cropped to make a square of then. Doing this, the quality loss was smaller in the resize of the image to fit 32x32. Without the crop, the accuracy of the classification was poor. 

Here are the results of the prediction:

| Image			        |     Prediction	        					
|:---------------------:|:-------------------------------------------------------:| 
| General caution       | General caution   							| 
| Right-of-way 			| Right-of-way 									|
| Road work				| Road work										|
| Speed limit (30km/h)	| Speed limit (30km/h)					 		|
| Stop					| Stop				 							|
| Ahead only			| Ahead only      								|


The model was able to correctly guess all the correct traffic signs, which gives an accuracy of 100%. I seams right because the images presented to the network were edited to make the classification easier. As the precision of the system is close to 94%, 100% in a 5 images data set is not a surprise. 

This is the print of the softmax of the predictions:

TopKV2(values=array([[  1.00000000e+00,   1.74748112e-35,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   4.84642201e-35,   1.21056489e-35,
          8.78199460e-36,   1.36707231e-38],
       [  9.99930978e-01,   6.90056550e-05,   3.46212037e-09,
          4.94936394e-13,   3.20212688e-13],
       [  9.68586624e-01,   3.14133763e-02,   2.79913426e-08,
          1.78823414e-11,   3.58998990e-14],
       [  1.00000000e+00,   4.33853121e-17,   6.50570273e-18,
          2.33617932e-18,   9.17762163e-19],
       [  1.00000000e+00,   1.12708916e-19,   1.27640310e-21,
          6.56199391e-22,   1.58750293e-22]], dtype=float32), indices=array([[18, 27,  0,  1,  2],
       [11, 42, 40, 21, 30],
       [25, 31, 29,  1, 22],
       [ 1,  0, 11,  6,  5],
       [14,  1,  3,  2, 25],
       [35, 36, 38, 41,  5]], dtype=int32))

It is possible to see that all the predictions are with very big certainty. The only prediction that has a relevant second option is the Speed limit (30km/h) guessing for the Speed limit (20km/h). This is reasonable as the images are a bit a like and this image is more distorted than the others.
