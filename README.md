# Stock Bot!

## Goal
Our goal was to create a machine learning model that could effectively predict stock prices on the stock market. We started with a linear regression model to make sure we could work with the simplest type of model and that the general format of our data was correct. We then quickly ramped up the complexity - we worked on implementing a neural network. After finally figuring out the format of the data that our network wanted, we began the slow and painful process of actually finding the right parameters and data structure that would let our network perform efficiently. At first, our network only predicted accurately on the first few days of data, but eventually over time we got a decent general prediction over the entire log of data and into the future. After this, we found an API for Google Search mentions and incorporated that into our network, and after some fine tuning, our model became very accurate on our data logs and the future. After testing this model on 29 of the largest publicly traded companies in the world, we found that on a good run our network could predict the one month return of 28 out of the 29 companies correctly, and have a 8.4% return compared to the 0.6% average return of the whole market.

## Documentation of Code

### `StockBot` class
Main stock bot function, saves a figure of the prediction of the trained neural network.

|function|arguments|purpose|
|---|---|---|
|constructor|company name, stock name, exchange|Initializes basic characteristics of object|
|pull|time of day (open, close, high, low, mid)|Read up to date stock data and Google trends data|
|scale|number of days from five years ago|Scale and reorganize the data into the input format for the NN|
|build|none|Construct and compile the neural network|
|train|epochs (iterations of training), verbose (1 or 0)|Train the neural network for the given number of epochs|
|predict|number of days from five years ago|Create a test data set and predict on it|
|graph|none|Create a plot of the current predicted data vs the actual stock data|

### Checking Script
Evaluates performance of the model.

### Prediction Script
Makes predictions on a set of stocks for how much each will return.

### Libraries Used
- `numpy`
  - Fast matrix/multidimensional array math library
- `matplotlib`
  - Creating beautiful plots
- `pandas_datareader`
  - Getting stock data
- `sklearn`
  - Data preprocessing
- `keras`
  - Constructing and training neural network
- `pytrends`
  - Getting google search trend data
- `time`
  - Timing
- `math`
  - Duh
