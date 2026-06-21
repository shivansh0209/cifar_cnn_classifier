##### In this file I write about the project journey of what I find new or what silly mistakes or valid mistakes I was doing while making the project.

### 1. Image Data Loading
There are two main ways which may be sufficint for small scale projects one is ImageDataGenerator(officially removed from keras) from keras.preprocessing and the other is the one is image_data_from_direct(). 

The ImageDataGenerator is used when you have a metadata tellinh which image name corresponds to which label so it loads the image on the fly while browsing through the metadata.csv. But as here we have used its alternative using keras we made a custom dataset and dataloader class. It can custom handle any type of situations just you need to change the functions.

Also to load single images to use them as samples or something the keras.preprocessing.image has a lot of things like load_image and load_img, img_to_array like functions. I used it when I wanted to pass a sample image to my submodel to see the trianed weights resulting in feature maps.

### 2. Python Lists
List(...) doesnt does work recursively it only unpacks the outermost layer but .toList() does it recursively to convert the data into a python list.


### 3. Interaction with layers in depth
```python
# Assume 'model' is your trained CNN

# 1. Print all layers and their shapes
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: Name={layer.name}, Type={type(layer).__name__}, Output Shape={layer.output_shape}")

# 2. Grab a specific layer by index or by name
first_conv_layer = model.layers[1] 
# Or: first_conv_layer = model.get_layer('conv2d_1')

# 3. Freeze a layer (Crucial for Transfer Learning)
# This tells model.fit() not to change this layer's weights during training
first_conv_layer.trainable = False

# 4. Extract raw weights and biases
# weights[0] will be the filters, weights[1] will be the biases
weights, biases = first_conv_layer.get_weights()
print("Filters shape:", weights.shape)

```

### 4. BatchNorm and biases
As BatchNorm uses bias so you can always disable the biases in the dense or convulation layers so as to save the memory. This hit me crazy and hard.

### 5. Crazy performance of BN
BN boosted th eaccuracy from 48 to 90% accuracy. As it solved the problem of internal covariate shift and dying relu. Dropout and BN solve differnt problems liek dropout(deccelerator) solve overfitting and BN solves underfitting(accelerator).

### 6. Custom Generalized CNN
I first time tried to make this model class and did a lot of silly mistakes one of the biggest was that in the call method we need to pass training object to the BN so that it can differentiate whether training is going on or not

### 7. Shape propagation
Reading theory is good but not very explaining everytime. When I built I understood what is happening . Suppose you gave an input of (16, 32, 32, 3) where 16 is the batch size . Now the conv2D layer has size of (3, 3). Note that although here it is written (3, 3) but the filter will always try to match the depth of the image so it will become (3, 3, 3) behind the hood . So then it interacts with image to always make a 2-D feature map. NOw this will be done by all the filters present and for one image all its corresponding feature maps get stacked on each other. Hence the shape now becomes (16, 32, 32, 32) if there were 32 filters . As it will happen for all 16 images . Now if the next conv layer has filter size of (3, 3) then it will behind the hood become (3, 3, 32) basically same filter will get stacked on each other 32 times then again it happens....

Pooling ALWAYS handle each feature map in a 2-d way and individukally , hence it only always effect the image width and hieght like form (16, 32, 32, 64) it becomes (16, 16, 16, 64).

Say after all feature extraction pooling etc we are left with 16, 8, 8, 128. After flattening there will be 8 * 8 * 128 features with us. ANd this batch size 16 will be untouched as it is denoting 1 tensor for each image.

Now you can assume it as a normal ANN input where a batch of 16 images with input size of 8 * 8 * 128 is entering.