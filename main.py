# File: main.py (Corrected with Lifespan Manager)

import os
import pickle
import re
import telegram
from telegram.constants import ParseMode
from fastapi import FastAPI, Request
from nltk.corpus import stopwords
from dotenv import load_dotenv
from contextlib import asynccontextmanager # <-- 1. IMPORT THIS

# --- Load Models and Preprocessing (remains the same) ---
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    exit()

TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telegram.Bot(token=TOKEN)

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return " ".join(words)

# --- 2. Create the Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("Application startup...")
    RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-hook"
        response = await bot.set_webhook(url=webhook_url)
        if response:
            print(f"Webhook set to {webhook_url}")
        else:
            print("Failed to set webhook")
    
    yield # The application runs while the server is alive
    
    # This code runs on shutdown (optional)
    print("Application shutdown...")


# --- 3. Initialize FastAPI with the Lifespan ---
app = FastAPI(lifespan=lifespan)

# --- 4. Webhook for the Telegram Bot (remains the same) ---
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
            
        await bot.send_message(chat_id=chat_id, text=reply_message, parse_mode=ParseMode.MARKDOWN)

    except KeyError:
        return {"status": "ignored"}
    
    return {"status": "ok"}