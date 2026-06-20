##### In this file I write about the project journey of what I find new or what silly mistakes or valid mistakes I was doing while making the project.

### 1. Image Data Loading
There are two main ways which may be sufficint for small scale projects one is ImageDataGenerator from keras.preprocessing and the other is the one which is used in this project. The one which I used in projects is mainly for the case when there images grouper into folders classwise. 

The ImageDataGenerator is used when you have a metadata tellinh which image name corresponds to which label so it loads the image on the fly while browsing through the metadata.csv

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