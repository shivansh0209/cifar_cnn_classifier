import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D
from keras.preprocessing.image import load_img, img_to_array
import math

def plot_metrics(history, title = "Model metrics"):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Validation Loss')
    plt.title(f"{title} - Validation Accuracy and Loss")
    plt.legend()
    plt.show()


def plot_feature_maps(model, conv_layer_number, input_image_path):
    # 1. Load and prepare the real image
    raw_image = load_img(input_image_path, target_size=(32, 32))
    image_array = img_to_array(raw_image)
    image_array = image_array.reshape((1, 32, 32, 3))

    # 3. Dynamically find the requested Conv2D layer
    target_layer = None
    counter = 0
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            counter += 1
            if counter == conv_layer_number:
                target_layer = layer
                break
                
    if target_layer is None:
        print(f"Error: The model only has {counter} Conv2D layers. You asked for layer {conv_layer_number}.")
        return

    # 4. Modern Fix: Use model.inputs (plural list) instead of model.input
    activation_function = Model(inputs=model.inputs, outputs=target_layer.output)
    feature_maps = activation_function.predict(image_array)

    no_of_feature_maps = feature_maps.shape[-1]

    # 5. Graphing grid parameters
    cols = 8
    rows = math.ceil(no_of_feature_maps / cols)

    plt.figure(figsize=(16, 2 * rows))

    for i in range(no_of_feature_maps):
        single_feature_map = feature_maps[0, :, :, i]
        
        ax = plt.subplot(rows, cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.imshow(single_feature_map, cmap='plasma')
        plt.title(f"Map {i+1}", fontsize=8)

    # 6. Render the canvas clean
    plt.tight_layout()
    plt.show()


__all__ = ["plot_metrics", "plot_feature_maps"]