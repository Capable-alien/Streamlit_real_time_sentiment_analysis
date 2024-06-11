import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load data
data = pd.read_csv(r'data\your_sentiment_data.csv', encoding='latin-1', header=None)
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data = data[['target', 'text']]

# Map target labels to binary sentiment (0=negative, 4=positive)
data['sentiment'] = data['target'].map({0: 'negative', 4: 'positive'})

# Define preprocessing and tokenization functions
def preprocess_text(text):
    return text

def tokenize_text(text):
    return text.split()

# Feature extraction and model training
vectorizer = TfidfVectorizer(preprocessor=preprocess_text, tokenizer=tokenize_text)
model = LogisticRegression(max_iter=1000)

# Pipeline for vectorization and model fitting
pipeline = make_pipeline(vectorizer, model)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model with cross-validation
cv_accuracy = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f'Cross-validated Accuracy: {cv_accuracy}')

# Evaluate model on test set
test_accuracy = pipeline.score(X_test, y_test)
print(f'Test Set Accuracy: {test_accuracy}')

# Save model and vectorizer
joblib.dump(pipeline, 'model/sentiment_pipeline.pkl')
