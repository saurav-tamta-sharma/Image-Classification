import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('cifar10_cnn_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to compute Grad-CAM
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    pooled_grads = tf.reshape(pooled_grads, [-1, 1, 1, pooled_grads.shape[-1]])
    conv_outputs *= pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]

    return heatmap

# Function to display Grad-CAM heatmap
def display_gradcam(img, heatmap, alpha=0.5):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Check the heatmap's shape
    if heatmap.ndim == 3:
        # Extract the first channel if necessary
        heatmap = heatmap[..., 0]

    # Convert to 8-bit format
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap
    superimposed_img = cv2.addWeighted(img, 1, heatmap, alpha, 0)
    return superimposed_img

# Function to classify the image
def classify_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return class_names[predicted_class], predictions

# Streamlit UI
st.title("Image Classifier with Grad-CAM")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', width=300)

    if st.button('Classify'):
        result, predictions = classify_image(img)
        st.write(f':green[Predicted Class:] {result}')

        # Get the Grad-CAM heatmap
        last_conv_layer_name = "conv2d"  # Change this to your model's last conv layer name
        heatmap = get_gradcam_heatmap(model, preprocess_image(img), last_conv_layer_name)

        # Display the Grad-CAM image
        gradcam_image = display_gradcam(img, heatmap)
        st.image(gradcam_image, caption='Grad-CAM', width=300)

        st.write("### Heatmap Legend")
        st.write("""
        - **Blue:** Low importance
        - **Green:** Medium importance
        - **Red:** High importance
        - **Yellow:** Very high importance
        """)