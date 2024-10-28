# python3.11 predict_pickle.py

import pickle

dv_file = 'dv.bin'
model_file = 'model1.bin'

with open(dv_file, 'rb') as f_in: 
    dv = pickle.load(f_in)
    
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    
customer = {"job": "management", "duration": 400, "poutcome": "success"}
X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)
