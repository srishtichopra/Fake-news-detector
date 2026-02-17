# Fake News Detector 🕵️‍♂️
A machine learning tool that uses a PassiveAggressive Classifier to detect misinformation in news articles.

## Setup
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your dataset as `data/news.csv` (CSV should have 'text' and 'label' columns).
4. Run the detector: `python app.py`

## How it works
It uses **TF-IDF Vectorization** to analyze linguistic patterns and a **PassiveAggressive Classifier** to categorize text as REAL or FAKE.
