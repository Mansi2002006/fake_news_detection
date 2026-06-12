import pickle
import os
import re
from flask import Flask, request, render_template
import wikipedia

app = Flask(__name__)

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

MIN_CONFIDENCE_PERCENT = 75.0
MIN_DECISION_PERCENT = 60.0
MIN_WORDS_FOR_STRONG_RESULT = 60

TRUSTED_SOURCES = [
    "reuters",
    "bbc",
    "cnn",
    "associated press",
    "ap news",
    "al jazeera",
    "the guardian",
    "new york times",
    "washington post",
    "times of india",
    "the hindu",
    "ndtv"
]

RECENT_HISTORY = []

SAMPLE_NEWS = {
    "real": "The Reserve Bank of India kept interest rates unchanged after its latest policy meeting. Officials said the decision was taken to maintain price stability while supporting economic growth.",
    
    "fake": "NASA confirmed that the moon will turn green tonight and everyone watching it will receive free internet for one year.",
    
    "ambiguous": "The government announced a new policy after a meeting on Friday."
}

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    print("Model loaded successfully.")

except Exception as e:
    print(f"STARTUP ERROR: {e}")
    model = None
    vectorizer = None

def clean_runtime_input(text):

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r'^.*?\(\s*Reuters\s*\)\s*-\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Keep numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()



def style_based_fallback(cleaned_text):

    real_signals = [
        "government",
        "official",
        "officials",
        "ministry",
        "court",
        "parliament",
        "reserve bank",
        "announced",
        "confirmed",
        "reported",
        "according to",
        "central bank",
        "spokesperson",
        "statement"
    ]

    fake_signals = [
        "shocking",
        "you won t believe",
        "secret plan",
        "miracle cure",
        "conspiracy",
        "viral claim",
        "deep state",
        "breaking bombshell",
        "aliens",
        "illuminati",
        "free internet"
    ]

    real_score = sum(
        1 for signal in real_signals
        if signal in cleaned_text
    )

    fake_score = sum(
        1 for signal in fake_signals
        if signal in cleaned_text
    )

    if real_score >= fake_score + 2:
        return (
            "REAL",
            "Hybrid check detected formal institutional reporting patterns."
        )

    if fake_score >= real_score + 1:
        return (
            "FAKE",
            "Hybrid check detected sensational or suspicious wording."
        )

    return None, None


def detect_category(cleaned_text):

    categories = {

        "Government": [
            "government",
            "president",
            "prime minister",
            "parliament",
            "ministry"
        ],

        "Finance": [
            "bank",
            "inflation",
            "market",
            "economy",
            "interest rate"
        ],

        "Health": [
            "hospital",
            "doctor",
            "vaccine",
            "medicine",
            "health"
        ],

        "Science": [
            "nasa",
            "space",
            "scientist",
            "research"
        ],

        "Security": [
            "military",
            "army",
            "security",
            "police",
            "border"
        ]
    }

    for category, words in categories.items():

        if any(word in cleaned_text for word in words):
            return category

    return "General"



def predict_news_label(text):

    cleaned_query = clean_runtime_input(text)

    word_count = len(cleaned_query.split())

    vec = vectorizer.transform([cleaned_query])

    raw_prediction = str(model.predict(vec)[0])

    try:

        probabilities = model.predict_proba(vec)[0]

        class_index = list(model.classes_).index(raw_prediction)

        confidence = round(
            probabilities[class_index] * 100,
            2
        )

        source_bonus = 0

        for source in TRUSTED_SOURCES:
            if source in cleaned_query:
               source_bonus += 5

        confidence = min(confidence + source_bonus, 100)

    except Exception:

        confidence = None

    # Low confidence handling
    if confidence is not None and confidence < MIN_DECISION_PERCENT:

        fallback_prediction, fallback_note = style_based_fallback(
            cleaned_query
        )

        if fallback_prediction:

            return (
                fallback_prediction,
                confidence,
                fallback_note
            )

        return (
            "UNCERTAIN",
            confidence,
            "Model confidence is too low. Please verify with trusted sources."
        )

    # Short input warning
    if word_count < MIN_WORDS_FOR_STRONG_RESULT:

        return (
            raw_prediction,
            confidence,
            "Short text detected. Paste a full article for better accuracy."
        )

    # Medium confidence warning
    if confidence is not None and confidence < MIN_CONFIDENCE_PERCENT:

        return (
            raw_prediction,
            confidence,
            "Prediction confidence is moderate. Cross-check recommended."
        )

    return raw_prediction, confidence, None


def check_india_president(news_text):

    try:

        summary = wikipedia.summary(
            "President of India",
            sentences=3
        ).lower()

        current_president = "droupadi murmu"

        if current_president in summary:

            if current_president in news_text.lower():

                return (
                    "REAL",
                    "Fact-check verified the current President of India."
                )

            else:

                return (
                    "FAKE",
                    "Fact-check mismatch detected regarding President of India."
                )

    except Exception as e:

        return None, f"Fact-check unavailable: {str(e)}"

    return None, None


@app.route("/", methods=["GET", "POST"])

def index():

    prediction = None
    confidence = None
    fact_check = None
    reliability_note = None
    submitted_text = ""
    category = None

    if request.method == "POST":

        text_input = request.form.get("news", "").strip()

        submitted_text = text_input

        if text_input:

            if model is None or vectorizer is None:

                return (
                    "Internal Server Error: model files missing.",
                    500
                )

            prediction, confidence, reliability_note = predict_news_label(
                text_input
            )

            category = detect_category(
                clean_runtime_input(text_input)
            )

            # Optional fact check
            if "president of india" in text_input.lower():

                fact_prediction, fact_check = check_india_president(
                    text_input
                )

                if fact_prediction:

                    prediction = fact_prediction
                    confidence = 100.0
                    reliability_note = None

            # Save history
            RECENT_HISTORY.insert(0, {

                "preview": text_input[:90] +
                ("..." if len(text_input) > 90 else ""),

                "prediction": prediction,

                "confidence": confidence,

                "category": category
            })

            del RECENT_HISTORY[3:]

    return render_template(

        "index.html",

        prediction=prediction,

        confidence=confidence,

        fact_check=fact_check,

        reliability_note=reliability_note,

        submitted_text=submitted_text,

        category=category,

        history=RECENT_HISTORY,

        sample_news=SAMPLE_NEWS
    )



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
