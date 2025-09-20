# File: main.py

import os
import pickle
import re
from fastapi import FastAPI, Form, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from nltk.corpus import stopwords
from dotenv import load_dotenv

# --- 1. Initialization ---
load_dotenv() # Load environment variables from .env file

# Initialize FastAPI app
app = FastAPI()

# Load the saved vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    print("Please run train_model.py first to generate these files.")
    exit()

# Initialize Twilio client
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)

# Initialize stopwords for preprocessing
stop_words = set(stopwords.words('english'))

# --- 2. Helper Function for Preprocessing ---
def preprocess_text(text):
    """Cleans and preprocesses a single text message."""
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return " ".join(words)

# --- 3. API Endpoint for Scam Analysis ---
@app.post("/analyse-message")
async def analyse_message(request: Request):
    """Analyzes a message to determine if it's a scam."""
    data = await request.json()
    message_text = data.get('message', '')

    if not message_text:
        return {"error": "Message text is required."}

    # Preprocess and transform the message
    processed_text = preprocess_text(message_text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])

    # Predict using the loaded model
    prediction = model.predict(vectorized_text)[0]
    is_scam = bool(prediction == 1) # Convert numpy.int64 to standard Python bool

    return {"is_scam": is_scam, "message": message_text}

# --- 4. Webhook for the WhatsApp Bot ---
@app.post("/whatsapp-hook")
async def whatsapp_hook(Body: str = Form(...)):
    """Handles incoming WhatsApp messages from Twilio."""
    response = MessagingResponse()
    user_message = Body

    # Preprocess and predict
    processed_text = preprocess_text(user_message)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0] # This gives 0 or 1
    prediction_probability = model.predict_proba(vectorized_text)[0] # This gives probabilities

    print(f"--- PREDICTION DETAILS ---")
    print(f"Raw Prediction (0=Safe, 1=Scam): {prediction}")
    print(f"Prediction Probabilities: {prediction_probability}")

    is_scam = bool(prediction == 1)
    # Craft the reply
    if is_scam:
        reply_message = (
            "⚠️ *Warning!* This message has characteristics of a potential scam. "
            "Please do not click any links, share personal information, or make payments."
        )
        # You can add logic to send an educational image here if you want
        # msg.media("https://your-server.com/scam_info_image.png")
    else:
        reply_message = (
            "✅ This message seems safe, but always remain cautious. "
            "Never share your PIN or passwords with anyone."
        )
    
    response.message(reply_message)
    return Response(content=str(response), media_type="application/xml")

# --- 5. Root endpoint for testing ---
@app.get("/")
def read_root():
    return {"message": "Digi-Kavach API is running!"}