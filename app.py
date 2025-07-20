from flask import Flask, request, jsonify, render_template
import random
import re
import numpy as np
import json
import os
import warnings
import logging
import nltk
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')

# TensorFlow setup
tf_log = tf.compat.v1.logging
warnings.filterwarnings('ignore')
tf_log.set_verbosity(tf_log.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class AdvancedChatbot:
    def __init__(self):
        self.intents = self.load_intents_from_json()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
        self.sentence_vectors = None
        self.keyword_map = defaultdict(list)
        self.acronym_responses = {
            'iot': "IoT (Internet of Things) connects physical devices to the internet for data exchange.",
            'ai': "AI (Artificial Intelligence) enables machines to perform human-like cognitive tasks.",
            'ml': "ML (Machine Learning) allows systems to learn patterns from data without explicit programming.",
            'nlp': "NLP (Natural Language Processing) enables computers to understand and process human language.",
            'os': "OS (Operating System) manages computer hardware and software resources.",
            'dbms': "DBMS (Database Management System) organizes and manages data efficiently."
        }
        self.context = defaultdict(dict)
        self.prepare_data()

    def load_intents_from_json(self):
        try:
            with open('chatbot_data.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
                print("Successfully loaded chatbot data from JSON file.")
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}. Using empty intents.")
            return {"intents": []}

    def prepare_data(self):
        self.patterns, self.responses, self.tags = [], [], []
        for intent in self.intents.get("intents", []):
            for pattern in intent.get("patterns", []):
                processed = self.advanced_preprocess(pattern)
                self.patterns.append(processed)
                self.tags.append(intent["tag"])
                self.responses.append(intent.get("responses", []))
                for word in processed.split():
                    if len(word) > 3:
                        self.keyword_map[word].append(intent["tag"])

        if self.patterns:
            self.vectorizer.fit(self.patterns)

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_vectors = self.model.encode(self.patterns) if self.patterns else None
            print("SentenceTransformer model loaded successfully.")
        except Exception as e:
            print(f"SentenceTransformer error: {e}")
            self.sentence_vectors = None

    def advanced_preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        try:
            words = nltk.word_tokenize(text)
        except LookupError:
            words = re.findall(r'\b\w+\b', text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    def get_response(self, message):
        message = message.lower().strip()

        # Check acronym shortcuts
        if message in self.acronym_responses:
            return self.acronym_responses[message]

        processed = self.advanced_preprocess(message)
        if not processed:
            return "I didn't quite get that. Could you try rephrasing your question?"

        match_idx, score = self.get_semantic_similarity(processed)

        if score > 0.6:
            return random.choice(self.responses[match_idx])

        # Fallback response
        fallback_responses = [
            "I'm not sure I understand. Try asking about:",
            "I'm still learning! You could ask me about:",
            "I didn't catch that. Here are some topics I can help with:"
        ]

        all_topics = [intent['tag'] for intent in self.intents.get('intents', []) if intent['tag'].lower() not in ['greeting', 'goodbye']]
        if all_topics:
            sample_topics = random.sample(all_topics, min(3, len(all_topics)))
            response = (
                f"{random.choice(fallback_responses)}\n"
                f"- {', '.join(sample_topics)}\n"
                "Or try rephrasing your question with more details."
            )
        else:
            response = "I'm sorry, I didn't understand that. Could you try asking differently?"

        return response

    def get_semantic_similarity(self, query):
        if self.sentence_vectors is None:
            return -1, 0
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.sentence_vectors)
            idx = np.argmax(similarities)
            return idx, similarities[0][idx]
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return -1, 0

# Create chatbot instance
chatbot = AdvancedChatbot()

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get-response", methods=["POST"])
def get_bot_response():
    user_message = request.json.get("message", "")
    bot_reply = chatbot.get_response(user_message)
    return jsonify({"reply": bot_reply})

# Main
if __name__ == "__main__":
    app.run(debug=True)
