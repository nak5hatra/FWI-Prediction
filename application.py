import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

appliacation = Flask(__name__)
app = appliacation

## import ridge regressor and standard scaler pickel

ridge_model =pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler =pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='POST':
        Temprature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled = standard_scaler.transform([[Temprature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        print(result)
        
        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')

if __name__ =="__main__":
    app.run(host="0.0.0.0")
    
    