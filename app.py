import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

class FakeNewsDetector:
    def __init__(self):
        self.model_path = 'models/model.pkl'
        self.vectorizer_path = 'models/vectorizer.pkl'
        self.model = None
        self.vectorizer = None

    def train(self, csv_path):
        print("--- Training Started ---")
        df = pd.read_csv(csv_path)
        # Ensure your CSV has 'text' and 'label' columns
        x_train, x_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=7
        )

        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = self.vectorizer.fit_transform(x_train)
        tfidf_test = self.vectorizer.transform(x_test)

        self.model = PassiveAggressiveClassifier(max_iter=50)
        self.model.fit(tfidf_train, y_train)

        # Evaluation
        y_pred = self.model.predict(tfidf_test)
        print(f'Accuracy: {round(accuracy_score(y_test, y_pred)*100, 2)}%')

        # Save to disk
        os.makedirs('models', exist_ok=True)
        pickle.dump(self.model, open(self.model_path, 'wb'))
        pickle.dump(self.vectorizer, open(self.vectorizer_path, 'wb'))
        print("Model saved to /models folder.")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.vectorizer = pickle.load(open(self.vectorizer_path, 'rb'))
            return True
        return False

    def predict(self, text):
        if not self.model:
            if not self.load_model():
                return "Error: No model found. Train it first!"
        
        tfidf_input = self.vectorizer.transform([text])
        return self.model.predict(tfidf_input)[0]

if __name__ == "__main__":
    detector = FakeNewsDetector()
    
    # Path to your dataset (e.g., from Kaggle's ISOT dataset)
    data_file = 'data/news.csv' 
    
    if not os.path.exists('models/model.pkl'):
        if os.path.exists(data_file):
            detector.train(data_file)
        else:
            print(f"Please place your dataset in {data_file}")
    
    # Quick Test
    news_input = input("Enter news text to verify: ")
    result = detector.predict(news_input)
    print(f"Result: This news is likely {result}")
