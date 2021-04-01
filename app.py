import numpy as np
import pandas as pd
from sklearn import preprocessing
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


final_data = pd.read_csv('final.csv')

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1),
    max_features =10000)

unigramdataGet= word_vectorizer.fit_transform(final_data['tweet_text'].astype('str'))
unigramdataGet = unigramdataGet.toarray()

vocab = word_vectorizer.get_feature_names()
unigramdata_features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
unigramdata_features[unigramdata_features>0] = 1

pro= preprocessing.LabelEncoder()
encpro=pro.fit_transform(final_data['class'])
final_data['class'] = encpro

y=final_data['class']
X=unigramdata_features





app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_value = request.form.values()
    x = word_vectorizer.transform(input_value)
    
    
    def prediction(value):
        result = model.predict(value)
        if result == 1:
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
        if result == 1:
            return 'Postive'
        else:
            return 'negative'

    output = prediction(x)
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)