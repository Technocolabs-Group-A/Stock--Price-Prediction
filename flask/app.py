import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('HistoricalData_APPLE.csv', index_col="Date", parse_dates=True)
data['Open'] = data['Open'].str.replace('$', '').astype(float)
data['Close/Last'] = data['Close/Last'].str.replace('$', '').astype(float)
data['High'] = data['High'].str.replace('$', '').astype(float)
data['Low'] = data['Low'].str.replace('$', '').astype(float)
y = data['Close/Last']
x = data.drop('Close/Last', axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
