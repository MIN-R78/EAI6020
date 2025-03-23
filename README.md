# Movie Recommendation System - Machine Learning Project

This project implements a movie recommendation system based on a neural network, which can predict the user's rating of a movie based on the user ID and the movie ID. The project includes data preprocessing, model training, evaluation, and deployment.

## Project Structure
project/
├── data/ # Dataset directory
│ └── u.data # Original data file
├── models/ # Model file directory
│ ├── movie_recommendation_model.h5 # Trained model
│ ├── user_encoder.joblib # User ID encoder
│ └── item_encoder.joblib # Movie ID encoder
├── app.py # Flask API service
├── test_api.py # API test script
├── train.py # Data preprocessing and model training script
└── README.md # Project description file

## Environment requirements

- Python 3.8+
- Dependent libraries:
- Flask
- TensorFlow
- scikit-learn
- pandas
- numpy


