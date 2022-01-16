from flask import Flask, render_template, url_for, request, redirect
import joblib

app = Flask(__name__)



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=["POST"])
def predict():
	import numpy as np 
	import pandas as pd 
	import matplotlib.pyplot as plt
	import joblib
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	age = request.form['age']
	sex = request.form['sex']
	bmi = request.form['bmi']
	child = request.form['children']
	smoker = request.form['smoker']
	region = request.form['region']
	sex_encoder = joblib.load("sex_encoder.sav")
	smoker_encoder = joblib.load("smoker_encoder.sav")
	region_encoder = joblib.load("region_encoder.sav")
	regression = joblib.load("regression.sav")
	predict_decoder = joblib.load("predict_decoder.sav")

	sex_e = sex_encoder.transform(np.array(sex).reshape(1,-1))
	smoker_e = smoker_encoder.transform(np.array(smoker).reshape(1,-1))
	region_e = region_encoder.transform(np.array(region).reshape(1,-1))

	predict = regression.predict(np.array([age,sex_e,bmi,child,smoker_e,region_e]).reshape(1,-1))
	predict1 = predict_decoder.inverse_transform(predict).round(2).tolist()
	return render_template('result.html', predict=predict1[0][0])

if __name__ == "__main__":
	app.run(debug=True)