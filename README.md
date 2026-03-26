# Vehicle Type Classification System 

## 📌 Objective
This project classifies vehicle images into:
- Car
- Bike
- Truck
- Bus
- Ambulance

## 📊 Dataset
- Source: Kaggle dataset
- Classes: 5
- Categories: Car, Bike, Bus, Truck, Ambulance
- Image format: JPG
- Note: Only sample images are uploaded in this repository

## ⚙️ Preprocessing
- Image resizing (224x224)
- Normalization
- Train and test split

## 🤖 Model
- Convolutional Neural Network (CNN)
- Trained using TensorFlow/Keras
- Provides prediction with confidence score

## How to Run
1. Install required libraries:
   pip install -r requirements.txt

2. Run the training:
   python train.py

3. Run the prediction:
   python app.py

## Output
- Predicts vehicle type
- Shows confidence level
- Shows decision
