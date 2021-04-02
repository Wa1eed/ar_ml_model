import numpy as np
import pandas as pd
from sklearn import preprocessing
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
word_vectorizer = pickle.load(open('vec_file.pickel', 'rb'))
pro = pickle.load(open('pro.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_value = request.form.values()
    x = word_vectorizer.transform(input_value)
    
    def prediction(value):
        result = model.predict(value)
        y = pro.inverse_transform(result)
        if y == 1:
            return 'Postive'
        else:
            return 'negative'


    output = prediction(x)

    return render_template('index.html', prediction_text='The Text is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    x = word_vectorizer.transform(data)
    
    def prediction(value):
        result = model.predict(value)
        y = pro.inverse_transform(result)
        if y == 1:
            return 'Postive'
        else:
            return 'negative'

    output = prediction(x)
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)