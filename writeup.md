## Project: Perception Pick & Place
## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points


[//]: # (Image References)

[objects]: ./images/objects.png
[table]: ./images/table.png
[clustering]: ./images/clustering.png

[confusion1]: ./images/confusion1.png
[confusion2]: ./images/confusion2.png
[confusion3]: ./images/confusion3.png

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
The pre-processing stage contains the below steps:

+ make_statistical_outlier_filter: remove noise which is points not belong to any object.
+ voxel_grid_downsampling: construct the less intensive cloud points which bring better calculation performance.
+ pass_through_filter: remove non-interesting area like below the table(pass_through_filter(points, 'z', 0.6, 1.1)) or left-right side of the table(pass_through_filter(points, 'y', -0.4, 0.5)). Only objects above the table are considered.
+ ransac_filter: detect table plane and separate table plane with object above.

After filtering and RANSAC, we could separate table and objects as images below:

![alt text][objects]
![alt text][table]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
Using euclidean clustering to group spatial related points together without knowing how many clusters in advance.
After that, label individual cluster's points with unique color and visualize it in screen.

![alt text][clustering]


#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Before enable robot to recognize the objects, we first need to generate training data and train recognition models.

1. Generate training data
Three set of training data(for three test) are generated with each contains 50 different poses. 
The color histogram feature is based on HSV which reduce the lighting condition effect on training data.
The remaining training feature is normal vectors which represent the surface orientation of the object.

2. Train generated data with SVM, there are three confusion matrix for three tests:

![alt text][confusion1]
![alt text][confusion2]
![alt text][confusion3]

3. Object recognition, using model generated at step 2, we can label objects from cloud points in real-time.
After clustering, individual cluster are recognized by the SVM model. Predicted labels are then published to the channel in order to be displayed above the object's position.

![alt text][objects]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

After trying some variant, I observed that, choosing the correct hyper parameters affect the performance of the tasks. For example:

1. For pass_through_filter, we must try different range(both z and y axis) to filter out non-interested data. Non-interested if exist will also be clustered and predicted then we have the false positives.
2. Generate training data, trying with only 10 iterations and the confusion matrix seems good. 
But when trying to use this model to predict objects, it is failed 30%. My assumption is because of under-fitting.
Trying with 50 iterations help to generate the better recognition models.

How to improve the project:

We could use deep learning instead of SVM, SVM is a linear learning algorithm so it is hard to predix complex objects.


