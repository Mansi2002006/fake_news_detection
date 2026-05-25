# Fake News Detector 2.0

An open-source web app that detects fake news headlines using machine learning and offers instant fact-checking for India’s President claims.

## Project Information

- **Author:** Mansi  
- **Roll No:** 12302039  
- **Institute:** University College of Engineering (UCOE), Punjabi University Patiala  
- **Supervisor:** Mr. Rahul  
- **Training Duration:** 9 June 2025 – 18 July 2025

## Features

- Classifies news headlines as `REAL` or `FAKE` using an ML model (Logistic Regression/SVM).
- Fact-checking for “President of India” claims using live Wikipedia summary.
- Easy web interface (Flask UI for local app, Gradio demo for Hugging Face Spaces).
- Open access for academic and educational use.

## How to Run Locally

1. **Clone the repository:**
    ```
    git clone https://github.com/yourusername/fake-news-detector.git
    cd fake-news-detector
    ```

2. **Setup and activate virtual environment:**
    ```
    python -m venv venv
    venv\Scripts\activate        # On Windows
    source venv/bin/activate    # On Linux/Mac
    ```

3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

4. **Run the web app:**
    ```
    python app.py
    ```
    Open browser to `http://127.0.0.1:5000`

## How to Use Hugging Face Spaces

- Try the public demo at: [Space link here after deployment]
- Enter a news headline, click Analyze, view result and fact-check.

## Example Inputs

