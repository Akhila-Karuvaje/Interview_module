from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
import speech_recognition as sr
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import json
import time
import logging
import requests

# === Extra imports for video evaluation ===
import cv2
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer, util
import whisper
import mediapipe as mp

# ============================================================
# ===== LOGGING CONFIGURATION =====
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# ===== NLTK DATA DOWNLOAD (Runtime) =====
# ============================================================
def download_nltk_data():
    """Download NLTK data at runtime to avoid Docker build issues"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already downloaded")
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")

download_nltk_data()

# ============================================================
# ===== FLASK APP CONFIGURATION =====
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret_key_change_in_production')

# ===== GROQ API CONFIGURATION using requests =====
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables!")
    raise ValueError("GROQ_API_KEY must be set in environment variables")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
logger.info("Groq API configured successfully")

# ============================================================
# ===== ML MODELS LOADING (with error handling) =====
# ============================================================
logger.info("Loading ML models...")

try:
    model_whisper = whisper.load_model("tiny")
    logger.info("Whisper model loaded successfully (tiny)")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model_whisper = None

try:
    model_bert = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("BERT model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load BERT model: {e}")
    model_bert = None

# ============================================================
# ===== HELPER FUNCTIONS =====
# ============================================================

def call_groq_api(prompt, max_retries=3):
    """
    Calls Groq API using direct HTTP requests with retry logic
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an expert interview assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API returned status {response.status_code}: {response.text}")
                raise Exception(f"API error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Groq API Error (Attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return json.dumps({
                    "error": "API temporarily unavailable. Please try again.",
                    "status": "failed"
                })
    
    return None

def SpeechToText():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source)
            logger.info("Listening...")
            audio = r.listen(source)
        logger.info("Recognizing...")
        query = r.recognize_google(audio, language='en-IN')
        return query
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clean_answer(answer):
    words = word_tokenize(answer)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word.lower() not in stop_words])

# ============================================================
# ===== ROUTES =====
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "whisper_loaded": model_whisper is not None,
        "bert_loaded": model_bert is not None
    }), 200

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    job = request.form['job']
    level = request.form['level']
    session['job_title'] = job
    session['difficulty'] = level
    logger.info(f"Generating questions for {job} - {level}")
    return redirect(url_for('regenerate_questions'))

@app.route('/regenerate_questions')
def regenerate_questions():
    job = session.get('job_title')
    level = session.get('difficulty')

    prompt = f"""
    Generate exactly 10 interview questions for the job role: {job} 
    with difficulty level: {level}. 
    Only return the 10 questions in plain text, numbered 1 to 10. 
    Do not include any introduction or extra comments.
    """
    
    response_text = call_groq_api(prompt)
    
    if not response_text or "error" in response_text.lower():
        logger.warning("API failed, using fallback questions")
        questions = [
            f"Tell me about your experience in {job}.",
            f"What are your key strengths for this {job} role?",
            f"Describe a challenging project you worked on.",
            f"How do you handle tight deadlines?",
            f"What motivates you in your career?",
            f"Where do you see yourself in 5 years?",
            f"How do you stay updated with industry trends?",
            f"Describe your ideal work environment.",
            f"What is your greatest professional achievement?",
            f"Why are you interested in this {job} position?"
        ]
    else:
        raw_questions = response_text.strip().split("\n")
        questions = []
        for q in raw_questions:
            match = re.match(r'^\d+[\).\s-]+(.*)', q.strip())
            if match:
                questions.append(match.group(1).strip())
        questions = questions[:10]
    
    session['questions'] = questions
    logger.info(f"Generated {len(questions)} questions")
    return redirect(url_for('questions'))

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    job = session.get('job_title')
    difficulty = session.get('difficulty')
    question_list = list(enumerate(questions, start=1))
    return render_template('questions.html', questions=question_list, job_title=job, difficulty=difficulty)

@app.route('/interview/<int:qid>')
def interview(qid):
    questions = session.get('questions', [])
    if 1 <= qid <= len(questions):
        question = questions[qid - 1]
    else:
        question = 'No question found'
    return render_template('interview.html', question=question, qid=qid)

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    
    audio_path = "/tmp/user_audio.wav"
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
        duration = 10
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition failed: {e}"}), 500
    except Exception as e:
        logger.error(f"Audio analysis error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    return jsonify({
        "transcription": transcribed_text,
        "duration": duration
    })

@app.route('/submit_answer/<qid>', methods=['POST'])
def submit_answer(qid):
    user_answer = request.form.get('answer', '').strip()

    prompt = f"""
    You are an expert interviewer. Analyze the user's answer concisely in JSON:

    {{
        "correct_answer": "Ideal answer",
        "validation": "Valid/Invalid/Partial",
        "feedback": "Brief feedback"
    }}

    Question ID: {qid}
    User Answer: "{user_answer}"
    """
    
    response_text = call_groq_api(prompt)
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON object found")
    except Exception as e:
        logger.error(f"JSON parsing error: {str(e)}")
        result = {
            "correct_answer": "Unable to parse response.",
            "validation": "Unknown",
            "feedback": "N/A"
        }

    return jsonify({
        'user_answer': user_answer,
        'validation_result': {
            'correct_answer': result.get('correct_answer', ''),
            'validation': result.get('validation', ''),
            'feedback': result.get('feedback', '')
        },
    })

@app.route('/video_interview')
def video_interview():
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    if not model_whisper:
        return jsonify({"error": "Whisper model not loaded"}), 500

    file = request.files['video']
    
    os.makedirs("/tmp/uploads", exist_ok=True)
    filepath = os.path.join("/tmp/uploads", f"answer_{qid}.webm")
    file.save(filepath)

    try:
        result = model_whisper.transcribe(filepath)
        transcript = result['text']
        
        if os.path.exists(filepath):
            os.remove(filepath)

        prompt = f"""
        You are an expert interview evaluator.
        Analyze this interview answer for question ID {qid} and return JSON in this exact format:
        {{
            "Confidence Score": <float between 0 and 1>,
            "Content Relevance": <float between 0 and 1>,
            "Fluency Score": <float between 0 and 1>,
            "Feedback": "3–5 line constructive feedback on how the user can improve"
        }}

        User's answer transcript: "{transcript}"
        """

        response_text = call_groq_api(prompt)
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group()
            scores = json.loads(json_str)
        else:
            scores = {
                "Confidence Score": 0.5,
                "Content Relevance": 0.5,
                "Fluency Score": 0.5,
                "Feedback": "Unable to analyze answer properly."
            }

        final_eval = round(
            (scores["Confidence Score"] +
             scores["Content Relevance"] +
             scores["Fluency Score"]) / 3 * 100, 2
        )

        session['video_feedback'] = scores["Feedback"]

        return jsonify({
            "Confidence Score": scores["Confidence Score"],
            "Content Relevance": scores["Content Relevance"],
            "Fluency Score": scores["Fluency Score"],
            "Final Evaluation": final_eval,
            "Transcript": transcript,
            "Feedback": scores["Feedback"]
        })
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

@app.route('/result')
def result():
    feedback = session.get('video_feedback', "No feedback available yet.")
    return render_template('result.html', feedback=feedback)

# ============================================================
# ===== ERROR HANDLERS =====
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {str(e)}")
    return render_template('500.html'), 500

# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
