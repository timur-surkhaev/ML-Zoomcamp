# sudo docker run --rm -it -p 0.0.0.0:8000:8000 test_01
# python3.11 predict_test_2.py


import requests

url = 'http://0.0.0.0:8000/predict'

client = {"job": "management", "duration": 400, "poutcome": "success"}

response = requests.post(url, json=client).json()
print(response)
