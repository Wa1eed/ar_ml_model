# Arabic Sentiment analysis using Machine Learning

The model was trained on local machine and been serialized using pickle to be used the in web app and reduce the computation . The REST API is built using Flask framework and all the details on how you send request mentioned below . Fun project and I will add more models to the same application in future release. 

## Installation

Using [git](https://github.com/Wa1eed/ar_ml_model.git) to install the model and the API on your local machine.

```bash
git clone https://github.com/Wa1eed/ar_ml_model.git

```

## Usage of api (Example)

```python
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Text':'حزين'}) 

print(r)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Date source to bulid the model 

https://www.kaggle.com/imranzaman5202/arabic-twitter-sentiment-analysis

## Model deatils 
Logistic Regression Algorithm
Accuracy= 0.766
Precision 0.76 %
Recall 0.73 %
F1 0.76 %



