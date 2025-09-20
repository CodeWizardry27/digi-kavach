# File: train_model.py (Final Version with MultinomialNB Model)

import pandas as pd
import re
#import nltk
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # <-- 1. IMPORT THE NEW MODEL
from sklearn.metrics import accuracy_score
import pickle

# --- 1. Load the Final Dataset ---
try:
    df = pd.read_csv('final_dataset.csv')
    df.columns = ['message', 'label'] 
    print("Successfully loaded final_dataset.csv.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# --- 2. Preprocessing ---
def preprocess_text(text):
    """Cleans text without removing stopwords."""
    text = str(text)
    # This line just removes special characters and converts to lowercase
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    return text

df['message'] = df['message'].apply(preprocess_text)
df = df.dropna()

# --- 3. Feature Extraction & Label Encoding ---
print("Performing feature extraction with N-grams...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['message']).toarray()
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# --- 4. Model Training with Stratified Split ---
print("Training the final model with MultinomialNB...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# ✅ 2. USE THE NEW MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# --- 5. Evaluation and Saving ---
print("Evaluating and saving the final model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ FINAL MODEL ACCURACY: {accuracy:.4f}")

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nFinal model and vectorizer saved successfully!")