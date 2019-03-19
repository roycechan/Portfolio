# Deep Thumbnail Face Classification and Verification
## Data
###  Size of datasets
Image Size = 3 x 32 x 32

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
In terms of model architecture, there are many existing model architectures to choose from, such as ResNet, Alex Net, VGG Net, MobileNet. 

For this implementation, I have decided to experiment with a highly efficient newer architecture - ShuffleNetV2. ShuffleNetV2 is a ResNet-like model that uses residual blocks, with the main innovations being the use of point-wise group convolutions (instead of normal group convolutions) and channel shuffle. 

**Basic idea of point-wise group convolution:** Combine the output channels of the depthwise convolution to create new features
![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/pointwise-conv.png)

**Basic idea of channel shuffle:** Enable cross-group information flow for multiple group convolution layers
![Channel Shuffle](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/channel_shuffle.JPG)

### Adjustments to existing architecture
Original ShuffleNetV2 architecture: 
![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/shufflenetv2_architecture.JPG)
ShuffleNetV2 was originally applied on ImageNet. In this regard, the input images in our dataset are much smaller i.e. 32 x 32 as compared to 224 x 224. Also, the number of output classes is higher i.e. 2300.

Hence, the following changes were made to the original architecture:

 - Conv1: Reduced stride from 2 to 1, reduced kernel size from 3 to 2
 - Maxpool: Removed
 - Stage3: Reduced repeats from 7 to 3
 - Stage4: Removed
 - Output channels: Increased as below

| Layer      | Output Size     | KSize | Stride | Repeat | Output Channels |
|------------|-----------------|-------|--------|--------|-----------------|
| Image      | 32 x 32         |       |        |        | 3               |
| Conv1      | 31 x 31         | 2x2   | 1      | 1      | 24              |
| Stage2     | 15 x 15 15 x 15 |       | 2 1    | 1 3    | 224             |
| Stage3     | 7 x 7 7 x 7     |       | 2 1    | 1 3    | 976             |
| Conv5      | 7 x 7           | 1x1   | 1      | 1      | 2048            |
| GlobalPool | 1 x 1           | 7x7   |        |        |                 |
| FC         |                 |       |        |        | 2300            |
 
### Code walkthrough
#### Data loading
Training and validation sets are loaded with ImageFolder

    train_dataset = torchvision.datasets.ImageFolder(root=paths.train_data_medium,
												    transform=train_transformations)
#### ShuffleNetV2
##### Basic block

![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/shufflenetv2_basic_block.JPG)

    class BasicUnit(nn.Module):  
	  def __init__(self, in_channels, splits=2, groups=2):  
		  super(BasicUnit, self).__init__()  
		  self.in_channels = in_channels  
		  self.splits = splits  
		  self.groups = groups  
	  
		  in_channels = int(in_channels / self.splits)  
		  self.right = nn.Sequential(*[  
						  nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),  
						  nn.BatchNorm2d(in_channels),  
						  nn.ReLU(inplace=True),  
						  nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels),  
						  nn.BatchNorm2d(in_channels),  
						  nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),  
						  nn.BatchNorm2d(in_channels),  
						  nn.ReLU(inplace=True)  
						 ])  
		  init_weights(self)  
	  
	  def forward(self, x):  
		  split = torch.split(x, int(self.in_channels / self.splits), dim=1)  
		  x_left, x_right = split  
	      x_right = self.right(x_right)  
		  x = torch.cat([x_left, x_right], dim=1)  
		  out = channel_shuffle(x, self.groups)  
		  return out
##### Down block
![](https://github.com/roycechan/Portfolio/blob/master/Deep%20Thumbnail%20Face%20Classification%20and%20Verification/resources/shufflenetv2_down_block.JPG)

    class DownUnit(nn.Module):  
	  def __init__(self, in_channels, out_channels, groups=2):  
		  super(DownUnit, self).__init__()  
		  self.in_channels = in_channels  
		  self.out_channels = out_channels  
		  self.groups = groups  
		  
		  self.left = nn.Sequential(*[  
					  nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, bias=False, groups=self.in_channels),  
					  nn.BatchNorm2d(self.in_channels),  
					  nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=1, stride=1, bias=False),  
					  nn.BatchNorm2d(self.out_channels // 2),  
					  nn.ReLU(inplace=True)  
					 ])  
		  self.right = nn.Sequential(*[  
					  nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, bias=False),  
					  nn.BatchNorm2d(self.in_channels),  
					  nn.ReLU(inplace=True),  
					  nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, bias=False, groups=self.in_channels),  
					  nn.BatchNorm2d(self.in_channels),  
					  nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=1, stride=1, bias=False),  
					  nn.BatchNorm2d(self.out_channels // 2),  
					  nn.ReLU(inplace=True)  
					 ])  
		  init_weights(self)  
	  
	  def forward(self, x):  
		  x_left = self.left(x)  
		  x_right = self.right(x)  
		  x = torch.cat([x_left, x_right], dim=1)  
		  out = channel_shuffle(x, self.groups)   
		  return out

##### Channel shuffle 
Given a convolutional layer with *g* groups whose output has *g × n* channels; we first reshape the output channel dimension into *(g, n)*, transposing and then flattening it back as the input of next layer. 

    def channel_shuffle(x, num_groups):  
		  N, C, H, W = x.size()  
		  x_reshape = x.reshape(N, num_groups, C // num_groups, H, W)  
		  x_permute = x_reshape.permute(0, 2, 1, 3, 4)  
		  return x_permute.reshape(N, C, H, W)

##### Forward pass
Forward pass is straight-forward. Note that we removed *maxpool* and *stage4* in our implementation. 

    def forward(self, x):
		out = self.conv1(x)
		out = self.stage2(out)
		out = self.stage3(out)
		# out = self.stage4(out)
		out = self.conv5(out)
		out = self.global_pool(out)
		out = out.view(out.size(0), -1) # flatten
		out = self.fc(out)
		return out

### Results
**Test Classification Accuracy: 78.23%**
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
