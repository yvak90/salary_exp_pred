# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:33:30 2020

@author: ajayy
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('salary_exp.pkl','rb'))

@app.route('/')
def home():
    return render_template('sal_exp_index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    yrs_exp = float_features[0];
    yrs_exp_sq = yrs_exp ** 2
    yrs_exp_cb = yrs_exp ** 3
    sal = np.exp(model.predict([[ yrs_exp, yrs_exp_sq, yrs_exp_cb ]]))
    print("Expected salary for {} experienced person is ${}".format(yrs_exp, int(sal[0][0]) ))

    return render_template('sal_exp_index.html', prediction_text= "Expected salary for {} years experienced person is ${}".format(yrs_exp,  int(sal[0][0]) ) )


if __name__ == "__main__":
    app.run(debug=True)