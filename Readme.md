
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
the dataset we considered is web scrapped from APIs.
The Historical Dataset came from NASDAQ API and News Articles are from Yahoo Finance

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
   Dropped features which are of no use the model. Removed outliers from data and make it more clean to use further.

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

# Data Modelling
So, after the exploratory data analysis we started modelling using Python.So for modelling we used Machine Learning algorithms on the datasets to build model to that will generate output for prediction of Stocks Price.In this step we have divided the data into train
and test as 80%,20% respectively. In this process we have used many
algorithms and applied some hyperparameter tuning so that our algorithms can
do better.
The algorithms which we have tried are:
1. Linear Regression
2. Naïve bayes
3. Neural networks


## LINEAR REGRESSION 
**Linear Regression** is a supervised learning algorithm in machine learning. It models a prediction value according to independent variables and helps in finding the relationship between those variables and the forecast and in this case we used last years dataset of companies to predict stocks value for future.

The accuracy score of model by linear regression</br>
**RMSE**(Root Mean Sqaured Error) = 0.1459830874093662</br>
**R-2**(R-Square Score) = 0.9998357614326422

![Screenshot (185)](https://user-images.githubusercontent.com/54480904/126519079-8a26eda3-f33d-4f6f-8b3b-2f231b6064c5.png)

![WhatsApp Image 2021-06-30 at 2 45 21 PM](https://user-images.githubusercontent.com/62774372/123957141-1215bb00-d9c9-11eb-9a09-08a714929a08.jpeg)

## Naïve Bayes

**Naïve bayes** is a probabilistic classifier, which means it predicts on the basis of the probability of an object. It is called <b> Naïve</b> because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. It is called <b>Bayes</b> because it depends on the principle of Bayes' Theorem.

 Bayes' Theorem -  &nbsp; ![naive-bayes-classifier-algorithm](https://user-images.githubusercontent.com/54480904/124004729-ee1c9e80-d9f5-11eb-9137-e6e10a6bf7e6.png)

Predicting the Impact of News articles on the Closed Price of the Apple Inc. Stocks using Naive Bayes Classifier. Firstoff all we merge the News Articles dataset and Historical Stocks Dataset into a single dataset on the 'Date' column after making some necessary changes to them. Now we add two more column named 'close_price_diff' and 'Impact' to the dataset, with 'close_price_diff' column containing the difference in Closed Price from the previous day and 'Impact' column containing 1 if the Closed Price difference is positive and 0 if it is negative. Afterwards we apply Natural Language Processing on the News Headlines text and obtain a Bag of words containing 20000 most common words from them by converting them to vectorized form. Now we train the Naive Bayes model (Gaussian, Multinomial or Bernoulli each in different files) by the splitting the dataset, 80% as training dataset and 20% as test dataset. Finally we do HyperParameter tuning to get the best predicted results.
We are classifying the news articles such that our model helps in classifying the news articles to be a profit or a loss.
We are doing this by calculating the diff in closed price of present day with the previous day.</br>
The Accuracy score in Naïve bayes is **51.93%** </br>
And After Hyperparameter Tuning it increased to **53.29%**

![Screenshot (186)](https://user-images.githubusercontent.com/54480904/126519775-b2e51bfd-43cf-4dd2-bea2-f78caef92baa.png)


![WhatsApp Image 2021-06-30 at 10 09 33 PM](https://user-images.githubusercontent.com/54480904/123999788-82840280-d9f0-11eb-8803-1317c108605e.jpeg)


## RNN LSTM
**Neural networks**, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. As the name suggest Neural network, it is quiet like our brain where there are some neurons working to get us the output. Then comes **RNN** which is a type of Neural Network which uses sequential data or time series data. **Long Short-Term Memory (LSTM)** networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.

The rmse score in LSTM is 101.3501

![Screenshot (187)](https://user-images.githubusercontent.com/54480904/126520705-81c74d0e-527e-4cc5-a219-d5b9ed125cbe.png)


![Screenshot (27)](https://user-images.githubusercontent.com/62774372/123957916-08d91e00-d9ca-11eb-8664-5b59067a4730.png)

## LSTM using 20 Days Data 

Predicting the closing stock price of a Apple Inc. using the past 20 day stock price by an artificial recurrent neural network called LSTM. We combine Historical data of Apple stocks prices and News articles data after some necessary changes to make them useful to get a combined dataset. Then we apply Sentiment Analysis on the News Headlines of the dataset to get 'compound', 'positive', 'negative' and 'neutral' values from it. After making some necessary changes and visualizing the data in various ways, we finalize 'close_price' and 'compound' as our features and 'close_price' as our dependent variable. The model is then trained on 80% of the data after applying the Feature Scaling (MinMaxScaler) on the features and tested on the remaining. We train the model by adding sufficient number of LSTM and Dense layers and using appropriate parameters values. At last the model predicts the values of 21st day Closed Price using past 20 days Closed Price and Compound value generated from the news headlines.

![Screenshot (184)](https://user-images.githubusercontent.com/54480904/126518777-5f3281ed-576d-4828-8c39-3a2ee7f018b9.png)


## Modeling And Deployment
 The model we choose finally is Linear Regression and Deployed it on heroku and streamlit. we used flask framework to upload model on website.


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
 
 ## Team Members
  1. Chandrachud Singh Chundawat
  2. V. Nanda Gopal
  3. Rahul Amarwal 
  4. Kondapu Lavanya
  5. Sunil Mali
  6. Sandeep Mannam
  7. Giduturi Namrata Sai
  8. Bale Meghana
  9. Sital Agrawal

### Team Leader 
   * Chandrachud Singh Chundawat

### Coordinator Name
   * Mr. Yasin shah
