from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

with open('model.pckl', 'rb') as model_file:
    Lrdetect_Model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text_input']
    prediction = Lrdetect_Model.predict([text_input])
    return render_template('index.html', language=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
