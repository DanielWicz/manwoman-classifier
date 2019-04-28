# Man-Woman Classifier
### By Daniel Wiczew

Woman and man image classifier based on convolutional neural network.
The architecture is a self-modified version of VGG16 [1] model.


### Example:
| <img src="example_man1.jpg?raw=true" width="200">|<img src="example_woman1.jpg?raw=true" width="200">|
|:-:|:-:|
|Man: 0.99846715<br/>Man: Woman 0.00153284|Man: 0.000028477365<br/>Woman: 0.99997151|

## Usage
### Preparation to run
Install dependencies from requirements.txt by executing pip command.

`pip install -r /path/to/requirements.txt`

### Using the predictor

Run the program `python main.py` and fallow the menu by choosing
1 and typing path of a jpg, png or bnp image, i.e. `data/Aman.jpg`.

### Architecture
The neural network is based on 19 layers, where 13 layers 
are convolutional layers with weights from VGG16 [1] model using transfer learning technique, 3 layers are 
full connected layers, with batch normalisation applied before every layer.

#### Model
<pre>
Leyer    (Type)                        Output dimensions
_________________________________________________________________
conv2d_1 (Convolutional 3x3)           (None, 224, 224, 64)      1792      
_________________________________________________________________
conv2d_2 (Convolutional 3x3)           (None, 224, 224, 64)      36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling 2x2)       (None, 112, 112, 64)      0         
_________________________________________________________________
conv2d_3 (Convolutional 3x3)           (None, 112, 112, 128)     73856     
_________________________________________________________________
conv2d_4 (Convolutional 3x3)           (None, 112, 112, 128)     147584    
_________________________________________________________________
max_pooling2d_2 (Max Pooling 2x2)      (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_5 (Convolutional 3x3)           (None, 56, 56, 256)       295168    
_________________________________________________________________
conv2d_6 (Convolutional 3x3)           (None, 56, 56, 256)       590080    
_________________________________________________________________
conv2d_7 (Convolutional 3x3)           (None, 56, 56, 256)       590080    
_________________________________________________________________
max_pooling2d_3 (MaxPooling 2x2)       (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_8 (Convolutional 3x3)           (None, 28, 28, 512)       1180160   
_________________________________________________________________
conv2d_9 (Convolutional 3x3)           (None, 28, 28, 512)       2359808   
_________________________________________________________________
conv2d_10 (Convolutional 3x3)          (None, 28, 28, 512)       2359808   
_________________________________________________________________
max_pooling2d_4 (MaxPooling 2x2)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv2d_11 (Convolutional 3x3)          (None, 14, 14, 512)       2359808   
_________________________________________________________________
conv2d_12 (Convolutional 3x3)          (None, 14, 14, 512)       2359808   
_________________________________________________________________
conv2d_13 (Convolutional 3x3)          (None, 14, 14, 512)       2359808   
_________________________________________________________________
max_pooling2d_5 (MaxPooling 2x2)       (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)                    (None, 25088)             0         
_________________________________________________________________
batch_norm_1 (Batch Normalization)     (None, 25088)             100352    
_________________________________________________________________
dense_1 (Fully Connected)              (None, 4096)              102764544 
_________________________________________________________________
batch_norm_2 (Batch Normalization)     (None, 4096)              16384     
_________________________________________________________________
dense_2 (Fully connected)              (None, 4096)              16781312  
_________________________________________________________________
batch_norm_3 (Batch Normalization)     (None, 4096)              16384     
_________________________________________________________________
dense_3 (Fully connected)              (None, 2)                 8194      
=================================================================
</pre>
As the optimizer Adam [2] was used and as the loss function Binary Crossentropy
#### Input
Input are images with shape (224, 224, 3), so they are 224 x 224 pixels with
3 RGB channels. Nevertheless, if you use other size or greyscale image, the
program will convert it the proper shape.

### Sensitivity and Specificity
The threshold is set to get the highest specificity and sensitivity.
It was set to 0.2 based on the ROC (Receiver operating characteristic)
and grid search. Giving sensitivity about 0.925 and specificity 0.956. 
#####ROC curve is shown below:
<img src="ROCcurve.jpg?raw=true" width="200">

### Dataset
The dataset was not published due to usage of own private images.

### Licence

### How to cite
D. Wiczew, Man-Woman Classifier (2019), GitHub repository `https://github.com/danielwicz/manwoman-classifier`

### References

[1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
[2] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

