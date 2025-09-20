# File: main.py (Corrected for Local Telegram Bot)

import os
import pickle
import re
import telegram
from telegram.constants import ParseMode # Correct import
from fastapi import FastAPI, Request
from nltk.corpus import stopwords
from dotenv import load_dotenv

# --- 1. Initialization ---
load_dotenv()
app = FastAPI()

# --- Load Models Locally ---
# This is the correct way for local testing. It loads files from your folder.
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    print("Please run train_model.py first to generate these files.")
    exit()

# --- Telegram Bot Setup ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telegram.Bot(token=TOKEN)

# --- Preprocessing Function (remains the same) ---
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return " ".join(words)

# --- 2. Webhook for the Telegram Bot ---
@app.post("/telegram-hook")
async def telegram_hook(request: Request):
    data = await request.json()
    
    try:
        chat_id = data['message']['chat']['id']
        user_message = data['message']['text']
        
        processed_text = preprocess_text(user_message)
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        is_scam = bool(prediction == 1)

        if is_scam:
            reply_message = "⚠️ *Warning!* This message looks like a potential scam."
        else:
            reply_message = "✅ This message seems safe, but always be cautious."
            
        # This block now uses the correctly imported ParseMode
        try:
           # The new, corrected line
            await bot.send_message(chat_id=chat_id, text=reply_message, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            print(f"--- ERROR SENDING MESSAGE: {e} ---")

    except KeyError:
        # This handles cases where the update is not a standard text message
        return {"status": "ignored"}
    
    return {"status": "ok"}