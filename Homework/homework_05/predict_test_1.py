# python3.11 predict.py
# python3.11 predict_test_1.py

import requests

#url = 'http://localhost:9696/predict'
url = 'http://127.0.0.1:8000/predict'


client = {"job": "student", "duration": 280, "poutcome": "failure"}

response = requests.post(url, json=client).json()
print(response)
