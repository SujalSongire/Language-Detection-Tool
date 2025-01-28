from flask import Flask, request, jsonify, render_template
import pickle
import os

# Ensure the template folder is correctly set
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load the model
with open('model.pckl', 'rb') as model_file:
    Lrdetect_Model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form.get('text_input', '')
    prediction = Lrdetect_Model.predict([text_input])
    return render_template('index.html', language=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
