# Stock Bot!

## Goal
Our goal was to create a machine learning model that could effectively predict stock prices on the stock market. We started with a linear regression model to make sure we could work with the simplest type of model and that the general format of our data was correct. We then quickly ramped up the complexity - we worked on implementing a neural network. After finally figuring out the format of the data that our network wanted, we began the slow and painful process of actually finding the right parameters and data structure that would let our network perform efficiently. At first, our network only predicted accurately on the first few days of data, but eventually over time we got a decent general prediction over the entire log of data and into the future. After this, we found an API for Google Search mentions and incorporated that into our network, and after some fine tuning, our model became very accurate on our data logs and the future. After testing this model on 29 of the largest publicly traded companies in the world, we found that on a good run our network could predict the one month return of 28 out of the 29 companies correctly, and have a 8.4% return compared to the 0.6% average return of the whole market.

## Documentation of Code

### `Stockbot` class
|function|arguments|purpose|
|---|---|---|
|pull|stock|pull data and store in obj|
|build|arguments relating to layers (list?)|resets network by rebuilding layers|
|train|range of dataset|builds a prediction function|
|predict|day|uses trained prediction function and scales|

### Testing Library
|function|arguments|purpose|
|---|---|---|
|simple_test|prediction function, |returns average difference b/t predicted value and actual value|

