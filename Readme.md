
# Stock Market Price Predictor using Supervised Learning
## Aim
To examine a number of different forecasting techniques to predict future stock returns based on past returns and numerical news indicators to construct a portfolio of multiple stocks in order to diversify the risk. We do this by applying supervised learning methods for stock price forecasting by interpreting the seemingly chaotic market data.
The fluctuation of the stock market is violent and there are many complicated financial indicators. However, the advancement in technology provides an opportunity to gain steady fortune from stock market and also can help experts to find out the ost informative indicators to make better prediction. The prediction of the market value is of paramount importance to help in maximizing the profit of stock option purchase while keeping the risk low. We have used previous datasets of stocks and news headines for the forecasting.

## Prerequisites

You need to have installed following softwares and libraries in your machine before running this project.

Python 3 Anaconda: It will install ipython notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy, scipy,streamlit.

## Libraries used

Pandas: For creating and manipulating dataframes.

Scikit Learn: For importing k-means clustering.

JSON: Library to handle JSON files.

XML: To separate data from presentation and XML stores data in plain text format.

Beautiful Soup and Requests: To scrap and library to handle http requests.

Matplotlib: Python Plotting Module.


  
## DATA

HistoricalData_APPLE.csv

![Screenshot](https://user-images.githubusercontent.com/62774372/123954864-86029400-d9c6-11eb-8981-bddf30c2ba56.png)


## Data Overview

<strong> Data Source </strong> --> Dataset/

<strong> Data points </strong>--> 2517 rows

<strong> Dataset date range </strong> --> October 2011 to September 2021

<strong> Dataset Attributes: </strong>

* Close/Last - Close/Last Prices

* Volume - Volume of Stocks

* Open - Opening Prices of Stocks

* High - Highest Prices of Stocks

* Low -  Lowest Prices of Stocks

## Data Preprocessing

## DATA CLEANING

    Deleted "Unnamed:7" Column For "Nan" Values Parsed The Date attribute in "datetime64" data type. Checked For Duplicate Rows(Not Found).

## EDA(Exploratry Data Analysis)

Exploratory Data Analysis is a process of examining or understanding the data and extracting insights or main characteristics of the data. EDA is generally classified into two methods, i.e. graphical analysis and non-graphical analysis.

Technically, The primary motive of EDA is to

    Examine the data distribution
    Handling missing values of the dataset(a most common issue with every dataset)
    Handling the outliers
    Removing duplicate data
    Encoding the categorical variables
    Normalizing and Scaling


### Here are some examples of data analysis we have done while exploring data

Data Visualization for all the columns for yearly wise

![Screenshot (23)](https://user-images.githubusercontent.com/62774372/123955329-1b9e2380-d9c7-11eb-839b-f54f9677c1f6.png)

Data Visualization for all the columns for monthly wise

![Screenshot (24)](https://user-images.githubusercontent.com/62774372/123955486-4a1bfe80-d9c7-11eb-9789-8d934598fbe5.png)

Data Visualization for all the columns for quarterly wise

![Screenshot (25)](https://user-images.githubusercontent.com/62774372/123955686-82234180-d9c7-11eb-9bd3-d5fdeb8a1cb7.png)


Scatter PLot is Plotted between each Attribute(Trend) 
![AyagWp9s8uRgAAAAAElFTkSuQmCC](https://user-images.githubusercontent.com/62774372/123956522-666c6b00-d9c8-11eb-9502-65c839802a2b.png)


Heat Matrix is Shown For Correlation Between Each Attribute(Linear Relation)
 ![Screenshot (19)](https://user-images.githubusercontent.com/62774372/123956726-a4698f00-d9c8-11eb-86ac-95ecd0987f3d.png)


## LINEAR REGRESSION 
**Linear Regression** is a supervised learning algorithm in machine learning. It models a prediction value according to independent variables and helps in finding the relationship between those variables and the forecast and in this case we used last years dataset of companies to predict stocks value for future.

The accuracy score of model by linear regression</br>
rmse(Root Mean Sqaured Error) = 0.1459830874093662</br>
r2(R-Square Score) = 0.9998357614326422


![WhatsApp Image 2021-06-30 at 2 45 21 PM](https://user-images.githubusercontent.com/62774372/123957141-1215bb00-d9c9-11eb-9a09-08a714929a08.jpeg)

## Naïve Bayes

**Naïve bayes** is a probabilistic classifier, which means it predicts on the basis of the probability of an object. It is called <b> Naïve</b> because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. It is called <b>Bayes</b> because it depends on the principle of Bayes' Theorem.

 Bayes' Theorem -  &nbsp; ![naive-bayes-classifier-algorithm](https://user-images.githubusercontent.com/54480904/124004729-ee1c9e80-d9f5-11eb-9137-e6e10a6bf7e6.png)

We are classifying the news articles such that our model helps in classifying the news articles to be a profit or a loss.
We are doing this by calculating the diff in closed price of present day with the previous day.</br>
The Accuracy score in Naïve bayes is **51.93%** </br>
And After Hyperparameter Tuning it increased to **53.29%**
![WhatsApp Image 2021-06-30 at 10 09 33 PM](https://user-images.githubusercontent.com/54480904/123999788-82840280-d9f0-11eb-8803-1317c108605e.jpeg)


## RNN LSTM
**Neural networks**, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. As the name suggest Neural network, it is quiet like our brain where there are some neurons working to get us the output. Then comes **RNN** which is a type of Neural Network which uses sequential data or time series data. **Long Short-Term Memory (LSTM)** networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.

The rmse score in LSTM is 101.3501

![Screenshot (27)](https://user-images.githubusercontent.com/62774372/123957916-08d91e00-d9ca-11eb-8664-5b59067a4730.png)




## Steps that we performed:

  * Web scrapped
  * Data Loading
  * Data Preprocessing
  * Exploratory data analysis
  * Feature engineering
  * Feature selection
  * Feature transformation
  * Model building
  * Model evalutaion
  * Model tuning
  * Prediction's

## Tools used:

  * Python
  * Pycharm
  * Jupyter Notebook
  * Google Colab
  * GitHub
  * GitBash
  * SublimeTextEditor 
