# Real-Time Sentiment Analysis Application

## Overview
This project is a real-time sentiment analysis application that analyzes the sentiment of user feedback in real-time. It allows users to input text, and the application provides feedback on whether the sentiment is positive or negative.

The application utilizes machine learning techniques, specifically logistic regression, for sentiment analysis. It is integrated with Amazon Web Services (AWS) for scalability and reliability, leveraging services such as AWS S3 for data storage and AWS DynamoDB for real-time data storage. During development and testing, LocalStack is used to emulate AWS services locally.

## Features
- Real-time sentiment analysis of user input text
- User-friendly web interface using Streamlit
- Integration with AWS S3 and DynamoDB for data storage
- Docker containerization for easy deployment
- Local development and testing using LocalStack

## Technical Details
- The sentiment analysis model is trained using scikit-learn and serialized using joblib.
- Data preprocessing and feature extraction are performed using TF-IDF vectorization.
- The application is containerized using Docker for easy deployment and management.
- Integration with AWS services is achieved using the Boto3 library.
- LocalStack is used to emulate AWS services locally for development and testing purposes.

## Usage
1. Install the required dependencies listed in `requirements.txt` using pip:
    pip install -r requirements.txt

2. Start LocalStack to emulate AWS services locally:
    localstack start

3. Run the Streamlit application using the following command:
    streamlit run app.py

4. Access the application in your web browser at the provided URL (default is http://localhost:8501).

5. Enter text in the input field and click on the "Analyze Sentiment" button to see the sentiment analysis result.

## Project Structure
- `app.py`: Streamlit web application for real-time sentiment analysis.
- `training.py`: Script for training the sentiment analysis model.
- `Dockerfile`: Dockerfile for containerizing the application.
- `requirements.txt`: List of Python dependencies required by the project.
- `data/`: Directory containing the dataset used for training the model.
- `model/`: Directory containing the trained sentiment analysis model.
