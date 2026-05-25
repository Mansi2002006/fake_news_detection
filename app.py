print("App.py is running")
from flask import Flask, request, render_template
import wikipedia
import pickle

def check_india_president(news_text):
    try:
        summary = wikipedia.summary("President of India", sentences=2)
        presidents = ["Droupadi Murmu", "Ram Nath Kovind"]
        for pres in presidents:
            if pres in summary:
                if pres.lower() in news_text.lower():
                    return f"Fact-check: TRUE ({pres} is current President)."
                elif "president of india" in news_text.lower():
                    return f"Fact-check: FALSE (The real President is {pres})."
        return "Could not confirm current president."
    except Exception as e:
        return "Fact-check error: " + str(e)

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ... (keep all imports and model loading as before)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None 
    fact_check = None 

    if request.method == "POST":
        text_input = request.form.get("news", "").strip()
        if text_input:
            vec = vectorizer.transform([text_input])
            pred = int(model.predict(vec)[0])
            label = "FAKE" if pred == 0 else "REAL"
            confidence = 100
            prediction = label
            # Only trigger fact-check if the news is about the president of India:
            if "president of india" in text_input.lower():
                fact_check = check_india_president(text_input)
            else:
                fact_check = None

    return render_template("index.html", prediction=prediction, confidence=confidence, fact_check=fact_check)

# ... (rest unchanged)

if __name__ == "__main__":
    print("Bottom of app.py reached, should start Flask now")
    app.run(debug=True)
