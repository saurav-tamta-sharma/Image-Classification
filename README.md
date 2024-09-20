**Image Classification using Tensorflow:**

This project demonstrates how to classify images using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. In addition to image classification, the project implements Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of the model's predictions. Grad-CAM highlights the regions of the input image that are most influential in the model's decision-making process, allowing users to understand why specific predictions are made.

The application features an intuitive interface built with Streamlit, where users can upload images for classification and visualize the corresponding Grad-CAM heatmap overlay.

**Features:**   
Classify Images: Classifies images into categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. Image Preview: Displays images from a local directory before classification.

**Prerequisites**:   
Python 3.x TensorFlow Matplotlib Pillow

*You can install the required packages using* pip: **pip install tensorflow matplotlib pillow**

**Model:**   
The model is a CNN trained on the CIFAR-10 dataset. The model file (cifar10\_with\_man\_model.h5) should be placed in the root directory of this project.  
