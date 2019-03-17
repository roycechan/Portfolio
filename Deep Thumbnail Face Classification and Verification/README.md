# Deep Thumbnail Face Classification and Verification
## Introduction

## Data
- size
- hierarchy
- samples

## Face Classification
### Introduction
Given a face image as input to your system, you have to predict the ID of the person's face. The true face image ID will be present in the training data and so the network will be doing an N-way classification to get the prediction. You are provided with the development set and can fine tune the model based on the accuracy you get on the development set
### Objectives
Evaluation metric: **Classification accuracy**
### Choosing a suitable architecture
![Channel Shuffle](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/channel_shuffle.JPG)
### Adjustments to existing architecture
### Code walkthrough
#### Data loading
#### ShuffleNetV2
##### Basic block
##### Down block
##### Channel shuffle 
##### Forward pass
### Results
## Face Verification
### Introduction
An input to your system is a trial, that is, a pair of face images that may or may not belong to the same person. Given a trial, your goal is to output a numeric score that quantifies how similar the images of the two people appear to be. On some scale, a higher score will indicate higher confidence that the two images belong to one and the same person in fact.
### Objectives
Evaluation metric: **AUC score**
### Code walkthrough
#### Data loading
#### Retrieve embeddings
#### Calculate similarity
### Results
## References
1. http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf
