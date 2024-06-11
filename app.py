import streamlit as st
import joblib
import boto3
import json
from datetime import datetime
import os

# Define preprocess_text function
def preprocess_text(text):
    # Add your preprocessing code here
    return text

# Define tokenize_text function
def tokenize_text(text):
    return text.split()

# Set dummy AWS credentials for LocalStack
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Function to check if a DynamoDB table exists
def does_table_exist(table_name):
    dynamodb = boto3.client('dynamodb', endpoint_url='http://localhost:4566', region_name='us-east-1')
    existing_tables = dynamodb.list_tables()['TableNames']
    return table_name in existing_tables

# Create S3 bucket if it does not exist
s3_client = boto3.client('s3', endpoint_url='http://localhost:4566', region_name='us-east-1')
if 'sentiment-bucket' not in s3_client.list_buckets()['Buckets']:
    s3_client.create_bucket(Bucket='sentiment-bucket')

# Create DynamoDB table if it does not exist
if not does_table_exist('SentimentScores'):
    dynamodb_client = boto3.client('dynamodb', endpoint_url='http://localhost:4566', region_name='us-east-1')
    dynamodb_client.create_table(
        TableName='SentimentScores',
        KeySchema=[
            {
                'AttributeName': 'Timestamp',
                'KeyType': 'HASH'
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'Timestamp',
                'AttributeType': 'S'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    )

# Load the trained pipeline
pipeline = joblib.load('model/sentiment_pipeline.pkl')

# Function to predict sentiment
def predict_sentiment(text):
    prediction = pipeline.predict([text])[0]
    return 'Positive' if prediction == 'positive' else 'Negative'

# Save results to S3 and DynamoDB
def save_results_to_s3_and_dynamodb(text, sentiment):
    timestamp = datetime.now().isoformat()
    data = {
        'text': text,
        'sentiment': sentiment,
        'timestamp': timestamp
    }
    
    # Save to S3
    s3_client.put_object(
        Bucket='sentiment-bucket',
        Key=f'sentiment_{timestamp}.json',
        Body=json.dumps(data)
    )
    
    # Create DynamoDB client here
    dynamodb_client = boto3.client('dynamodb', endpoint_url='http://localhost:4566', region_name='us-east-1')
    
    # Save to DynamoDB
    dynamodb_client.put_item(
        TableName='SentimentScores',
        Item={
            'Timestamp': {'S': timestamp},
            'Text': {'S': text},
            'Sentiment': {'S': sentiment}
        }
    )


# Streamlit app layout
st.title("Real-Time Sentiment Analysis")

user_input = st.text_area("Enter Text Here:")

if st.button("Analyze Sentiment"):
    sentiment = predict_sentiment(user_input)
    save_results_to_s3_and_dynamodb(user_input, sentiment)
    st.write(f"Sentiment: {sentiment}")
