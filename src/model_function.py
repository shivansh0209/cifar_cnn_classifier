from keras import layers, Model, Input

def FullyGeneralizedCNN(input_shape=(32, 32, 3), num_classes=10, use_batch_norm=True, use_max_pooling=True, strides=(1, 1), padding="same"):
    
    inputs = Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    
    # -------------------------------------------------------------------------
    # Block 1: Double Convolution (Learns low-level features like edges)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(32, (3, 3), strides=strides, padding=padding, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding=padding, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if use_max_pooling: x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x) # Light dropout early on
    
    # -------------------------------------------------------------------------
    # Block 2: Double Convolution (Learns mid-level features like shapes)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(64, (3, 3), strides=strides, padding=padding, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding=padding, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if use_max_pooling: x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x) # Increasing dropout
    
    # -------------------------------------------------------------------------
    # Block 3: Double Convolution (Learns high-level semantic features)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(128, (3, 3), strides=strides, padding=padding, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding=padding, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if use_max_pooling: x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x) # Heavy dropout before the classifier
        
    # -------------------------------------------------------------------------
    # Classifier Head (Reverting to Flatten safely)
    # -------------------------------------------------------------------------
    # With 3 pools on a 32x32 image, the shape here is 4x4. 
    # Flattening 4x4x128 = 2048 elements. This is small enough that Flatten 
    # is perfectly safe and often performs better than GAP for small datasets.
    x = layers.Flatten()(x) 
    
    x = layers.Dense(128, use_bias=not use_batch_norm)(x)
    if use_batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="FullyGeneralizedCNN")

__all__ = ["FullyGeneralizedCNN"]