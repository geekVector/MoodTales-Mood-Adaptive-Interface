from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import joblib
import re
import os
import ssl
import nltk
import requests
import time
import io
from gtts import gTTS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# 1. APP CONFIG
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# 2. LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_mode_lg.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
    print(" Sentiment model loaded successfully.")
except Exception as e:
    print(f" Model loading failed: {e}")
    pipeline = None

# -------------------------------
# 3. NLTK SETUP
# -------------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

nltk_data_path = os.path.join(BASE_DIR, "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# -------------------------------
# 4. TEXT PREPROCESSING
# -------------------------------
lemmatizer = WordNetLemmatizer()
default_stopwords = set(stopwords.words('english'))

sentiment_words = {
    "not", "no", "nor", "never",
    "don't", "didn't", "isn't", "wasn't",
    "won't", "can't", "couldn't"
}

custom_stopwords = default_stopwords - sentiment_words


def preprocess_text(text):
    text = re.sub(r'https?:\/\/\S+|www\.\S+|@\w+|<@!?[\d]+>', '', text)
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [
        word for word in tokens
        if (word.isalpha() or word in {"n't", "not"})
        and word not in custom_stopwords
    ]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# -------------------------------
# 5. HUGGINGFACE / LLAMA CONFIG
# -------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # 🔴 Replace with your HuggingFace token
API_URL = "https://router.huggingface.co/v1/chat/completions"

hf_headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


def generate_ai_story(sentiment: str, story_length: int = 120) -> str:
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct:fastest",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"""You are a master storyteller known for subverting expectations and crafting narratives 
                    that feel unlike anything the reader has encountered before.

                    Write a story in exactly {story_length} words.
                    Mood: {sentiment}.

                    STRICT RULES:
                    - Open with a sentence that is impossible to ignore — no weather, no waking up, no 'It was'
                    - Choose ONE unexpected narrative angle: second-person dread, unreliable objects as narrators, 
                    time told backwards, or a mundane setting hiding something cosmically wrong
                    - Every sentence must earn its place — no filler, no throat-clearing
                    - Build tension through *specificity*, not vagueness — name things, give details that feel 
                  uncomfortably real
                    - The twist must recontextualize the ENTIRE story, not just surprise at the end
                    - The final sentence should land like a punch and linger after the reader looks away

                    FORBIDDEN:
                    - Clichéd openings (dark stormy nights, alarm clocks, mirror reflections)
                    - Generic character names like John, Sarah, or 'the man'
                    - Neat, happy resolutions
                    - Telling the reader how to feel ('terrifyingly', 'sadly', 'shocking')

                    Mood filter — apply these to {sentiment}:
                    - If dark/horror: use cold, clinical language where warmth should exist
                    - If sad: find beauty inside the loss, not just grief
                    - If hopeful: let one small shadow remain at the edge
                    - If joyful: make the joy feel slightly too fragile to hold

                    Deliver only the story. No title. No commentary."""

                )
            }
        ],
        "max_tokens": story_length * 2,
        "temperature": 0.9
    }

    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=hf_headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                story = data["choices"][0]["message"]["content"].strip()
                if story:
                    return story
                print("⚠️ Empty story in response.")

            elif response.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"⏳ Model loading, retrying in {wait}s...")
                time.sleep(wait)
                continue

            elif response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"⏳ Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue

            else:
                print(f"❌ HF error {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout on attempt {attempt + 1}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        time.sleep(5)

    return "Sorry, I couldn't generate a story right now. Please try again."


# -------------------------------
# 6. ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ──────────────────────────────────────────────────────────────
# FEATURE 1 — SLIDER LENGTH
# The frontend sends { mood: "...", story_length: <int> }.
# story_length comes from the range slider (min 50, max 500).
# It is passed directly to generate_ai_story() as the word target.
# ──────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get("mood", "").strip()
        story_length = int(data.get("story_length", 50))  # ← from slider (Feature 1)

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        if not pipeline:
            return jsonify({"error": "Model not loaded"}), 500

        # Preprocess
        cleaned = preprocess_text(user_input)

        # Predict sentiment
        prediction = pipeline.predict([cleaned])[0]
        sentiment = "Sad" if prediction == 0 else "Happy"

        # Generate story using Llama via HuggingFace
        story = generate_ai_story(sentiment, story_length)

        return jsonify({
            "sentiment": sentiment,
            "story": story
        })

    except Exception as e:
        print("Server Error:", e)
        return jsonify({"error": "Internal server error"}), 500


# -----------------------------------------------------------------------
# FEATURE 2 — MIC / VOICE INPUT  (server-side fallback route)
#
# Primary path  → browser uses webkitSpeechRecognition (Web Speech API)
#                 to transcribe locally, then sends the text to /predict
#                 just like a typed message. No server change needed.
#
# Fallback path → POST raw audio blob to /transcribe if the browser
#                 doesn't support the Web Speech API (some mobile).
#                 Returns { "text": "<transcribed string>" }.
#                 Install dependency:  pip install SpeechRecognition
# -----------------------------------------------------------------------
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Accepts an audio file upload (WAV/WebM) and returns transcribed text.

    Frontend usage:
        const formData = new FormData();
        formData.append("audio", blob, "voice.wav");
        const res = await fetch("/transcribe", { method: "POST", body: formData });
        const { text } = await res.json();
        // feed `text` to /predict as { mood: text, story_length: N }
    """
    try:
        import speech_recognition as sr
        import tempfile

        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        suffix = '.wav' if audio_file.filename.endswith('.wav') else '.webm'

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        os.remove(tmp_path)

        return jsonify({"text": text})

    except ImportError:
        return jsonify({
            "error": "SpeechRecognition not installed. Run: pip install SpeechRecognition"
        }), 500
    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


# -----------------------------------------------------------------------
# FEATURE 3 — TEXT-TO-SPEECH NARRATION
#
# Accepts: POST { "story": "<story text>" }
# Returns: MP3 audio stream of the story narrated naturally via gTTS.
# Install dependency: pip install gTTS
# -----------------------------------------------------------------------
@app.route('/narrate', methods=['POST'])
def narrate():
    try:
        data = request.get_json()
        story_text = data.get("story", "").strip()

        if not story_text:
            return jsonify({"error": "No story text provided"}), 400

        # Generate MP3 audio in memory using gTTS (natural English narration)
        tts = gTTS(text=story_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='story_narration.mp3'
        )

    except Exception as e:
        print(f"Narration error: {e}")
        return jsonify({"error": f"Narration failed: {str(e)}"}), 500


# -------------------------------
# 7. RUN SERVER
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)