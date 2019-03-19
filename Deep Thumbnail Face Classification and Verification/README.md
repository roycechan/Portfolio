# Deep Thumbnail Face Classification and Verification
## Data
###  Size of datasets
||Classification  | Verification
|------|------------|----|
|  Training  | 822,000 images (2300 Face ids) | Same as Classification|
|Validation| 5,000 images| 100,000 image pairs
|Testing| 4,500 images	| 900,000 image pairs	
### Data hierarchy
![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/data_hierarchy.JPG)
### Samples
![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/random_photo_train_dataset.JPG)

## Face Classification
### Introduction
Predict the ID of a person's face given a face image as input. The true face image ID is present in the training data and so the network will be doing an N-way classification to get the prediction. This is a closed set problem.
![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/classification_example.jpg)

### Objectives
Evaluation metric: **Classification accuracy**
### Choosing a suitable architecture

### Adjustments to existing architecture
### Code walkthrough
#### Data loading
#### ShuffleNetV2
##### Basic block
##### Down block
##### Channel shuffle 
![Channel Shuffle](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/channel_shuffle.JPG)
##### Forward pass
### Results
## Face Verification
### Introduction
An input to your system is a trial, that is, a pair of face images that may or may not belong to the same person. Given a trial, your goal is to output a numeric score that quantifies how similar the images of the two people appear to be. On some scale, a higher score will indicate higher confidence that the two images belong to one and the same person in fact.

![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/verification_example.jpg)

### Objectives
Evaluation metric: **AUC score**
### Code walkthrough
#### Data loading
#### Retrieve embeddings
#### Calculate similarity
### Results
## References
1. http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf
2. https://arxiv.org/pdf/1807.11164.pdf
