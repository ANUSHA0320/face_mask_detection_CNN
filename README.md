developed a Face Mask Detection model that predicts bounding box coordinates around faces in images using a custom-trained Convolutional Neural Network (CNN).


**Tech Stack:**
TensorFlow & Keras for CNN-based bounding box regression
PIL for image processing
Pascal VOC XML annotations for training
Streamlit for an interactive web app interface

**How It Works:**
Trained a CNN model on a dataset containing face mask images and annotations. The model learns to predict bounding boxes of faces wearing or not wearing masks.


**Built a Streamlit web app that:**
Accepts an uploaded image
Predicts the bounding box
Labels it as "With Mask" or "Without Mask"
