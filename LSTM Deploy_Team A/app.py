import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import keras
import datetime

app = Flask(__name__)
model = keras.models.load_model("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

	stock_data = pd.read_csv('stock_data.csv', parse_dates = ['Date'], index_col = 'Date')
	X = stock_data['Close/Last']

	# Taking Date input
	val = 0
	for value in request.form.values():
		val = value
		break
	input_date = datetime.datetime.strptime(val + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

	# Getting the start day and next day from the dataset
	start_day = stock_data.index[0]
	last_day = stock_data.index[-1]
	next_day = last_day + datetime.timedelta(days = 1)

	if input_date <= next_day and input_date >= start_day + datetime.timedelta(days = 20):

		scaler = MinMaxScaler(feature_range=(0,1))

		# Create a list of dates from the stock_data and get the index of the input date
		dates_list = []
		for dt in stock_data.index:
			dates_list.append(str(dt))

		i = dates_list.index(str(input_date - datetime.timedelta(days = 1)))

		X =stock_data.filter(['Close/Last'])
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
		# Get the predicted scaled price
		pred_price = model.predict(X_test)
		# undo the scaling
		pred_price = scaler.inverse_transform(pred_price)
		# print('Predicted Close Price for {} '.format(next_day) + ' = ', pred_price)

		# Percentage increase or decrease in Closed Price
		previous = pred_price
		previous_pred_price = X.at[str(input_date - datetime.timedelta(days = 1)), 'Close/Last']

		diff=(float)(pred_price - previous_pred_price)
		if(diff < 0):
		  return render_template('index.html', prediction_text = 'Close Price for {} is ${}'.format(next_day, pred_price), percentage_diff = 'Percentage decrease = {}%'.format(round(((-(diff)/previous_pred_price)*100),2)))
		else:
		  return render_template('index.html', prediction_text = 'Close Price for {} is ${}'.format(next_day, pred_price), percentage_diff = 'Percentage increase = {}%'.format(round((((diff)/previous_pred_price)*100),2)))

	else:
		return render_template('index.html', prediction_text = 'Error: Either the date is above the last date of the dataset OR below the start date + 20 days of the dataset. Please enter a date between or equal to {} and {} !!'.format(start_day + datetime.timedelta(days = 20), next_day))

if __name__ == '__main__':
    app.run(debug=True)