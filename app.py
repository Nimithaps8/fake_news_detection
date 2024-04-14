from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load your vectorizer
vectorization = TfidfVectorizer()

# Load your trained SVM model from pickle file
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Define preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Define function for prediction and output labeling
def predict_news(news):
    new_x_test = vectorization.transform([news])
    y_pred = svm_model.predict(new_x_test)
    return "Fake news" if y_pred[0] == 0 else "Not A Fake News"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        prediction = predict_news(news)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

