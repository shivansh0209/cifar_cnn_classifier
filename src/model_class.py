from keras import layers, Model

class FullyGeneralizedCNN(Model):
    def __init__(self, num_classes=10, use_batch_norm=True, use_max_pooling=False, strides=(1, 1), padding="valid"):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_max_pooling = use_max_pooling
        
        # 0. Input Scaler
        self.rescaling = layers.Rescaling(1./255)
        
        # 1. Block 1 Layers
        self.conv1 = layers.Conv2D(32, (3, 3), strides=strides, padding=padding, use_bias=not use_batch_norm)
        if self.use_batch_norm:
            self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        if self.use_max_pooling:
            self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        
        # 2. Block 2 Layers
        self.conv2 = layers.Conv2D(64, (3, 3), strides=strides, padding=padding, use_bias=not use_batch_norm)
        if self.use_batch_norm:
            self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        if self.use_max_pooling:
            self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        
        # 3. Block 3 Layers
        self.conv3 = layers.Conv2D(128, (3, 3), strides=strides, padding=padding, use_bias=not use_batch_norm)
        if self.use_batch_norm:
            self.bn3 = layers.BatchNormalization()
        self.act3 = layers.ReLU()
        if self.use_max_pooling:
            self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
            
        # 4. Classifier Head Layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, use_bias=not use_batch_norm)
        if self.use_batch_norm:
            self.bn_dense = layers.BatchNormalization()
        self.act_dense = layers.ReLU()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    # CRITICAL: Added training=None here
    def call(self, inputs, training=None):
        x = self.rescaling(inputs)
        
        # --- Block 1 Execution ---
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x, training=training) # Pass training flag
        x = self.act1(x)
        if self.use_max_pooling:
            x = self.pool1(x)
            
        # --- Block 2 Execution ---
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training) # Pass training flag
        x = self.act2(x)
        if self.use_max_pooling:
            x = self.pool2(x)
            
        # --- Block 3 Execution ---
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x, training=training) # Pass training flag
        x = self.act3(x)
        if self.use_max_pooling:
            x = self.pool3(x)
            
        # --- Classifier Head Execution ---
        x = self.flatten(x)
        x = self.dense1(x)
        if self.use_batch_norm:
            x = self.bn_dense(x, training=training) # Pass training flag
        x = self.act_dense(x)
        
        return self.classifier(x)