# Automated Bone Fracture Detection using Deep Learning

This project is a smart medical system that detects bone fractures from X-ray images using deep learning techniques. It helps doctors by quickly and accurately identifying whether a bone is fractured and what type of fracture it is.

## Team Members
- Vydeti Sruthi Sree  
- Shaik Ameena  
- Issuru Shiva Shankar Reddy  
- Ponnamalla Rahul

## Objective
To build an automated system using Convolutional Neural Networks (CNNs) that:
- Detects fractures from X-ray images
- Classifies the type of fracture
- Reduces human error and speeds up diagnosis

## Technologies Used
- Python  
- TensorFlow, Keras  
- OpenCV  
- Flask (for web interface)  
- Google Colab (for training)  

## How It Works
1. User uploads an X-ray image via a simple web app.
2. The system preprocesses the image using filters and edge detection.
3. A trained CNN model analyzes the image.
4. It predicts if the bone is fractured or not and shows the result.

## How to Run
1. Clone this repository.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
