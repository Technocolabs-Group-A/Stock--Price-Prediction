
# Stock Market Price Predictor using Supervised Learning
# Aim
To examine a number of different forecasting techniques to predict future stock returns based on past returns and numerical news indicators to construct a portfolio of multiple stocks in order to diversify the risk. We do this by applying supervised learning methods for stock price forecasting by interpreting the seemingly chaotic market data.

## Prerequisites

You need to have installed following softwares and libraries in your machine before running this project.

Python 3 Anaconda: It will install ipython notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy, scipy,streamlit.

  
## DATA

HistoricalData_APPLE.csv

![Screenshot](https://user-images.githubusercontent.com/62774372/123954864-86029400-d9c6-11eb-8981-bddf30c2ba56.png)


## Data Overview

Data Source --> Dataset/

Data points --> 2517 rows

Dataset date range --> October 2011 to September 2021

Dataset Attributes:

Close/Last - Close/Last Prices

Volume - Volume of Stocks

Open - Opening Prices of Stocks

High - High Prices of Stocks

Low -  Low Prices of Stocks

## Data Preprocessing

## DATA CLEANING

    Deleted "Unnamed:7" Column For "Nan" Values Parsed The Date attribute in "datetime64" data type. Checked For Duplicate Rows(Not Found).

## EDA(Exploratry Data Analysis)

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

Linear_regression(2).ipynb

![WhatsApp Image 2021-06-30 at 2 45 21 PM](https://user-images.githubusercontent.com/62774372/123957141-1215bb00-d9c9-11eb-9a09-08a714929a08.jpeg)





## Steps that we performed:

   Web scrapped
   Data Loading
   Data Preprocessing
   Exploratory data analysis
   Feature engineering
   Feature selection
   Feature transformation
   Model building
   Model evalutaion
   Model tuning
   Prediction's

## Tools used:

   Python
   Pycharm
   Jupyter Notebook
   Google Colab
   GitHub
   GitBash
   SublimeTextEditor 
