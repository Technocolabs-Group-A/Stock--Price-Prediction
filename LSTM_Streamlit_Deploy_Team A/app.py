# -*- coding: utf-8 -*-
"""
Created on Fri July 20 12:50:04 2021
@author: Chandra chud singh chundawat
"""


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import keras
import datetime

model = keras.models.load_model("model.h5")

def welcome():
    return "Welcome All"

def predict_price(final_features):
	pred_price = model.predict(final_features)
	return pred_price

def main():
	st.title("Apple Inc. Stock Price Prediction")
	html_temp = """
	<div style="background-color:rgb(0, 238, 255);padding:10px">
	<h2 style="color:rgb(255, 124, 37);text-shadow: 0 4px 10px rgba(0, 0, 0, 0.603);text-align:center;">Apple Inc. Predicted CLosed Price</h2>
	</div>
	"""

	st.markdown(html_temp,unsafe_allow_html=True)
    
	stock_data = pd.read_csv('stock_data.csv', parse_dates = ['Date'], index_col = 'Date')
	X = stock_data['Close/Last']

	# Getting the start day and next day from the dataset
	start_day = stock_data.index[0]
	last_day = stock_data.index[-1]
	next_day = last_day + datetime.timedelta(days = 1)

	# Taking date input
	input_date = st.date_input("Enter a Date: ", next_day)
	# Updating Date input
	input_date = datetime.datetime.strptime(str(input_date) + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

	if input_date <= next_day and input_date >= start_day + datetime.timedelta(days = 20):

		scaler = MinMaxScaler(feature_range=(0,1))

		# Create a list of dates from the stock_data and get the index of the input date
		dates_list = []
		for dt in stock_data.index:
			dates_list.append(str(dt))

		j = 1
		while str(input_date - datetime.timedelta(days = j)) not in dates_list:
			j += 1

		i = dates_list.index(str(input_date - datetime.timedelta(days = j)))

		X = stock_data.filter(['Close/Last'])
		# Get the last 20 day closing price values and convert the dataframe to an array
		last_20_days = X[i-20: i].values
		# Scale the data to be values between 0 and 1
		last_20_days_scaled = scaler.fit_transform(last_20_days)
		# Create an empty list
		X_test = []
		# Append the past 20 days
		X_test.append(last_20_days_scaled)
		# Convert the X_test data set to a numpy array
		X_test = np.array(X_test)
		# Reshape the data
		X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

		# Predict the Close Price
		result = 0
		if st.button("Predict"):
			result = predict_price(X_test)

		# undo the scaling
		result = np.array(result).reshape(1,-1)
		pred_price = scaler.inverse_transform(result)

		st.success("Predicted Close Price for {} is ${}".format(input_date, pred_price))

		# Percentage increase or decrease in Closed Price
		previous = pred_price
		previous_pred_price = X.at[str(input_date - datetime.timedelta(days = j)), 'Close/Last']

		diff=(float)(pred_price - previous_pred_price)
		if(diff < 0):
			st.write("percentage decrease = ",round(((- (diff)/previous_pred_price)*100),2))
		else:
			st.write("percentage increase = ",round((( (diff)/previous_pred_price)*100),2))

	else:
		st.error('Error: Either the date is above the last date of the dataset OR below the start date + 20 days of the dataset. Please enter a date between or equal to {} and {} !!'.format(start_day + datetime.timedelta(days = 20), next_day))


if __name__ == '__main__':
    main()