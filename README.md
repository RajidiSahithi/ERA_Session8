#  CLASSIFING CIFAR10 DATASET USING RESNET

# CONTENTS 
- [DATASET](#dataset)
- [IMPORTING_LIBRARIES](#importing_libraries)
- [SET_THE_TRANSFORMS](#set_the_transforms)
- [SET_DATA_LOADER](#set_data_loader)
- [CNN_MODEL](#cnn_model)
- [TRAINING_THE_MODEL](training_the_model)


## DATASET 
### CIFAR DATASET
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color (RGB) images containing one of 10 object classes, with 6000 images per class.

## IMPORTING_LIBRARIES
Import the required libraries. 
* NumPy is used for for numerical operations.The torch library is used to import Pytorch.
* Pytorch has an nn component that is used for the abstraction of machine learning operations. 
* The torchvision library is used so that to import the CIFAR-10 dataset. This library has many image datasets and is widely used for research. The transforms can be imported to resize the image to equal size for all the images. 
* The optim is used train the neural Networks.
* MATLAB libraries are imported to plot the graphs and arrange the figures with labelling

## SET_THE_TRANSFORMS
Transforms are applied dataset, in this 
* We Normalize the image tensors by substracting the mean and dividing by the standard deviation across each channel. it will mean the data across each channel to 0 and standard deviation to 1. The mean and standard deviation of CIFAR10 Dataset is (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) respectively (3 channels - RGB)
* Normalization is applied for test dataset also  train the model using normalized data then model no longer understands the original pixel values

### DATA AUGMENTATION
Applying chosen transformations while loading images from the training dataset. Image is filiped horizontly with a 50% probability. Since the transformation will be applied randomly and dynamically each time a particular image is loaded, the model sees slightly different images in each epoch of training, which allows it generalize better. Along with flip Randomaffine is applied on training dataset for transulating, rotating and shaering the image


## SET_DATA_LOADER
* Training transforms applied on complete dataset.
* The dataset is set by dividing (randomly)  into 2 parts train_set,validation_set.
* train_set and test_set are totally different. 
* Here length of train set  =  40000
* length of  validation set =  10000
* Batch size is set to 128, Number of CPUs = 2
* train_loader is loaded with transformed train set. Validation_laoder loaded with transformed validation set
* Test set is also taken from test transformes and teat loader is taken
* Training dataset is completely diffrent from testing dataset

#### SAMPLE DATASET
![alt text](https://github.com/RajidiSahithi/ERA_Session8/blob/main/ImagesS8/Sample%20dataset.png)

## CNN_MODEL
#### RESNET CNN MODEL IS USED
The network has following layers
<pre>
Input >                      3X32X32
    
    Conv1(3X3) (normalization,ReLU,Dropout)     >     
        
        Conv23 > Conv2(3X3) (normalization,ReLU,Dropout)  + Conv3(1X1) (normalization,ReLU,Dropout)   
        (For this the receptive fields of Conv2 and Conv3 must be same, Output channels of Conv1,Conv2 and Conv3 must be same)
            
            MaxPool > Maxpooling with stride 2X2 and Filter Size (2X2)   >
                
                Conv3(3X3) (normalization,ReLU,Dropout) > 
                    
                    Conv4(3X3) (normalization,ReLU,Dropout)   >
                        
                        Conv56 > Conv5(3X3) (normalization,ReLU,Dropout)  + Conv5(1X1) (normalization,ReLU,Dropout)
                        (For this the receptive fields of Conv5 and Conv6 must be same, Output channels of Conv4,
                        Conv5 and Conv6 must be same)
                            
                            MaxPool > Maxpooling with stride 2X2 and Filter Size (2X2)  >
                                
                                Conv7(3X3) (normalization,ReLU,Dropout) > 
                                    
                                    Conv8(3X3) (normalization,ReLU,Dropout)   >
                                        
                                        Conv9(3X3) (normalization,ReLU,Dropout) > 
                                            
                                            Adaptive Average Pooling ()   >
                                                
                                                Conv10(1X1) Similar to fully connected layer with 10 output channels


            
</pre>
Dropout is 0.01 i.e., 1%
#### Receptive Field Calculation Sheet and Skeleton of CNN Model.
![alt text](https://github.com/RajidiSahithi/ERA_Session8/blob/main/ImagesS8/RF%20Calculation.png)
#### Parameter Count of the CNN model

<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 30, 30]             216
       BatchNorm2d-2            [-1, 8, 30, 30]              16
              ReLU-3            [-1, 8, 30, 30]               0
           Dropout-4            [-1, 8, 30, 30]               0
            Conv2d-5            [-1, 8, 30, 30]             576
       BatchNorm2d-6            [-1, 8, 30, 30]              16
              ReLU-7            [-1, 8, 30, 30]               0
           Dropout-8            [-1, 8, 30, 30]               0
            Conv2d-9            [-1, 8, 30, 30]              64
      BatchNorm2d-10            [-1, 8, 30, 30]              16
             ReLU-11            [-1, 8, 30, 30]               0
          Dropout-12            [-1, 8, 30, 30]               0
        MaxPool2d-13            [-1, 8, 15, 15]               0
           Conv2d-14           [-1, 16, 13, 13]           1,152
      BatchNorm2d-15           [-1, 16, 13, 13]              32
             ReLU-16           [-1, 16, 13, 13]               0
          Dropout-17           [-1, 16, 13, 13]               0
           Conv2d-18           [-1, 32, 11, 11]           4,608
      BatchNorm2d-19           [-1, 32, 11, 11]              64
             ReLU-20           [-1, 32, 11, 11]               0
          Dropout-21           [-1, 32, 11, 11]               0
           Conv2d-22           [-1, 32, 11, 11]           9,216
      BatchNorm2d-23           [-1, 32, 11, 11]              64
             ReLU-24           [-1, 32, 11, 11]               0
          Dropout-25           [-1, 32, 11, 11]               0
           Conv2d-26           [-1, 32, 11, 11]           1,024
      BatchNorm2d-27           [-1, 32, 11, 11]              64
             ReLU-28           [-1, 32, 11, 11]               0
          Dropout-29           [-1, 32, 11, 11]               0
        MaxPool2d-30             [-1, 32, 5, 5]               0
           Conv2d-31             [-1, 16, 3, 3]           4,608
      BatchNorm2d-32             [-1, 16, 3, 3]              32
             ReLU-33             [-1, 16, 3, 3]               0
          Dropout-34             [-1, 16, 3, 3]               0
           Conv2d-35             [-1, 16, 3, 3]           2,304
      BatchNorm2d-36             [-1, 16, 3, 3]              32
             ReLU-37             [-1, 16, 3, 3]               0
          Dropout-38             [-1, 16, 3, 3]               0
           Conv2d-39              [-1, 8, 1, 1]           1,152
      BatchNorm2d-40              [-1, 8, 1, 1]              16
             ReLU-41              [-1, 8, 1, 1]               0
          Dropout-42              [-1, 8, 1, 1]               0
AdaptiveAvgPool2d-43              [-1, 8, 1, 1]               0
           Conv2d-44             [-1, 10, 1, 1]              80
================================================================
Total params: 25,352
Trainable params: 25,352
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.13
Params size (MB): 0.10
Estimated Total Size (MB): 1.23
----------------------------------------------------------------
</pre>
Same is the parameter Count for Group Normalization and Layer Normalization

The model is observed by applying 3 Normalization Techniques
* Batch Normalization
* Group Normalization
* Layer Normalization

##### BATCH NORMALIZATION (BN)
* Batch Normalization focuses on standardizing the inputs to any particular layer(i.e. activations from previous layers). Standardizing the inputs mean that inputs to any layer in the network should have approximately zero mean and unit variance. Mathematically, BN layer transforms each input in the current mini-batch by subtracting the input mean in the current mini-batch and dividing it by the standard deviation.

* But each layer doesn’t need to expect inputs with zero mean and unit variance, but instead, probably the model might perform better with some other mean and variance. Hence the BN layer also introduces two learnable parameters γ and β.
* BN restricts the distribution of the input data to any particular layer(i.e. the activations from the previous layer) in the network, which helps the network to produce better gradients for weights update. Hence BN often provides a much stable and accelerated training regime.


##### GROUP NORMALIZATION (GN)
* Group Normalization is also applied along the feature direction , it divides the features into certain groups and normalizes each group separately. In practice, Group normalization performs better than layer normalization, and its parameter num_groups is tuned as a hyperparameter.
*  In GN each layer is divide into 2 groups (for this model). Always Number of Channels must be divisible by number of groups
* In this model it is observed that as group size is incresed accuracy is reduced


##### LAYER NORMALIZATION (LN)
* Layer normalization is same as group normalization with group size set to 1
* It is observed that LN accuracy is more compared to GN




## TRAINING_THE_MODEL
<pre>
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)
</pre>

Above Optimizers and Scheduler are used for training the model and back propagation is done for improving the accuracy.
* Learrning Rate = 0.01

#### Adam Optimizer 

Adaptive Moment Estimation is an algorithm for optimization techniques for gradient descent. The method is really efficient when working with significant problem involving a lot of data or parameters. It requires less memory and is efficient. Intuitively, it is a combination of the ‘gradient descent with momentum algorithm and the ‘RMSP’ algorithm. I observed that it worked better than SGD

#### ReduceLROnPlateau
Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. There are many strategies for varying the learning rate during training. In this model 
* I used "ReduceOnPlateue" and "One Cycle Learning Rate Policy", which involves starting with a low learning rate, gradually increasing it batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value for the remaining epochs.
* Weight decay: here I selected weight decay=1e-05. that is L2 (Ridge regularization) adds a squared magnitude of coefficient as a penalty term to the loss function.


###### OBSERVATIONS
* keep on running the model accuracy of the model is being increased
<pre>
Number of epochs=20
</pre>

#### Plot of Accuracy and Loss

![alt text](https://github.com/RajidiSahithi/ERA_Session8/blob/main/ImagesS8/acc_loss.png)https://github.com/RajidiSahithi/ERA_Session8/blob/main/ImagesS8/acc_loss.png


#### Misclassified Images

  ![alt text](https://github.com/RajidiSahithi/ERA_Session8/blob/main/ImagesS8/misclassified%20images.png)https://github.com/RajidiSahithi/ERA_Session8/blob/main/ImagesS8/misclassified%20images.png)




