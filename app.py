from flask import Flask, render_template, request, flash, jsonify, send_from_directory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

model = None
tokenizer = None

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

@app.before_first_request
def before_first_request():
    load_keras_model()
    load_tokenizer()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        text = request.form.get("user_text")
        sentiment = call_analyzer(text) # VADER results
        sentiment["custom model positive"] = sentiment_analysis(text)
        flash(sentiment)
    return render_template('form.html')

@app.route("/sentiment_analyzer", methods=["GET", "POST"])
def sentiment_analyzer():
    # TODO: Write the code that calls the sentiment analysis functions here.
    # hint: use request.method == "POST"
    if request.method == 'POST':
        text = request.form.get("user_text")
        sentiment = call_analyzer(text) # VADER results
        sentiment["custom model positive"] = sentiment_analysis(text)
        flash(sentiment)
    return render_template('form.html')

def call_analyzer(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    return sentiment

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]),2)


if __name__ == "__main__":
    app.run()