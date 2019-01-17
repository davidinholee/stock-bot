---
title: Machine Learning and Market Analysis
author:
- David Lee
- Max Katz-Christy
geometry: margin=1in
toc: true
toc-depth: 1
---

\newpage

# Proposal

We will be spending the semester researching and studying the complex field of machine learning. Machine learning is a way for computers to "learn" with data and be able to predict future outcomes or tendencies. For instance, creating an AI that can play chess, showing curated online ads, facial recognition, and Google Translate are all examples of where machine learning is applied. Our goal is to apply machine learning to various markets; we thought this would be interesting because markets are not very easily predictable by humans but may have predictable trends that a machine could pick up.

Our main project will be to analyze the stock market. We want to create a bot that will be able to predict whether stocks will rise or fall, given the past history of the stock as well as other data (like twitter mentions) that we need to determine. Our reasoning for starting with the stock market is that there are large backlogs of data that we can train and test our models on. We can also invest a ton of money after finishing our bot and get rich! If we are successful in accomplishing this task, we also have plans to apply similar systems to the cryptocurrency market or the job market.

By working together on this project, we will be able to combine the different skill sets we bring from our areas of "expertise". David has worked a lot with machine learning and the theory behind many of the concepts we will deal with through different internships in the past. Max has worked a lot with robotics and has a very well rounded computer science background, as well as currently taking an economics course at Harvard. We are both excited to work on this project for this current semester!

\newpage

# Brief Project Overview

Our goal was to create a machine learning model that could effectively predict stock prices on the stock market. We started with a linear regression model to make sure we could work with the simplest type of model and that the general format of our data was correct. We then quickly ramped up the complexity - we worked on implementing a neural network. After finally figuring out the format of the data that our network wanted, we began the slow and painful process of actually finding the right parameters and data structure that would let our network perform efficiently. At first, our network only predicted accurately on the first few days of data, but eventually over time we got a decent general prediction over the entire log of data and into the future. After this, we found an API for Google Search mentions and incorporated that into our network, and after some fine tuning, our model became very accurate on our data logs and the future. After testing this model on 29 of the largest publicly traded companies in the world, we found that on a good run our network could predict the one month return of 28 out of the 29 companies correctly, and have a 8.4% return compared to the 0.6% average return of the whole market.

# Supporting Documents

All of our work is documented and collected in the following github repository:
https://github.com/davidinholee/stock-bot/

\newpage

# Annotated Bibliography

+-----------------+-----------------+-----------------+-----------------+
| Citations       | Brief Summary   | Uses and limits | New Questions   |
|                 |                 | of this work in |                 |
|                 |                 | relation to our |                 |
|                 |                 | own thinking    |                 |
+-----------------+-----------------+-----------------+-----------------+
| 1\. Chollet,    | This book       | The code        | Is Keras, the   |
| Francois.       | offers an in    | examples        | main Python     |
| Deep Learning   | depth           | presented in    | library we are  |
| with Python.    | introduction to | the book use    | working with,   |
| Manning,        | deep learning   | the deep        | modularized to  |
| 2018.           | in Python.      | learning        | be able to have |
|                 | There are many  | framework Keras | all the         |
|                 | practical,      | which is built  | functionalities |
|                 | hands-on        | on top of       | of TensorFlow?  |
|                 | explorations of | Google\'s       | Keras basically |
|                 | machine         | TensorFlow      | makes           |
|                 | learning        | backend engine. | TensorFlow much |
|                 | concepts with   | We are going to | easier to read  |
|                 | code snippets   | use these same  | and learn (by   |
|                 | you can test    | packages for    | having easier   |
|                 | for yourself.   | our             | syntax), so it  |
|                 |                 | exploration, so | could be        |
|                 |                 | this book       | possible there  |
|                 |                 | really helps us | are some        |
|                 |                 | to understand   | functions that  |
|                 |                 | the basic       | are not         |
|                 |                 | syntax and      | directly        |
|                 |                 | starting        | transferable    |
|                 |                 | concepts that   | between the     |
|                 |                 | we need to      | two.            |
|                 |                 | understand.     |                 |
|                 |                 | However, much   |                 |
|                 |                 | of the book is  |                 |
|                 |                 | also the many   |                 |
|                 |                 | applications of |                 |
|                 |                 | deep learning   |                 |
|                 |                 | from computer   |                 |
|                 |                 | vision to       |                 |
|                 |                 | natural         |                 |
|                 |                 | language        |                 |
|                 |                 | processing,     |                 |
|                 |                 | which is        |                 |
|                 |                 | interesting to  |                 |
|                 |                 | learn about,    |                 |
|                 |                 | but most of it  |                 |
|                 |                 | is not          |                 |
|                 |                 | useful/relevant |                 |
|                 |                 | for our         |                 |
|                 |                 | specific        |                 |
|                 |                 | research.       |                 |
+-----------------+-----------------+-----------------+-----------------+
| 2\. Géron,      | This book also  | Instead of      | Can we use the  |
| Aurélien.       | offer an in     | using Keras,    | concepts that   |
| Hands-On        | depth           | this book       | are applied to  |
| Machine         | introduction to | mainly works    | these famous    |
| Learning with   | deep learning   | with the Python | datasets for    |
| Scikit-Learn    | in Python. Like | package         | the problem     |
| and             | Deep Learning   | scikit-learn,   | that we are     |
| TensorFlow      | with Python,    | which is a more | dealing with?   |
| Concepts,       | this book also  | "basic" form of | Although these  |
| Tools, and      | offers many     | TensorFlow      | datasets mostly |
| Techniques to   | good examples   | because it      | deal with       |
| Build           | and code        | cannot          | images or more  |
| Intelligent     | snippets, but   | construct       | mundane types   |
| Systems.        | uses some of    | neural          | of data, many   |
| O\'Reilly,      | the more        | networks. This  | of the big      |
| 2018.           | fundamental     | is very useful  | ideas can be    |
|                 | problems in     | to us because   | applied across  |
|                 | machine         | many of the     | the board to    |
|                 | learning.       | normalization   | almost any      |
|                 |                 | and overfitting | machine         |
|                 |                 | techniques we   | learning        |
|                 |                 | have to use are | problem?        |
|                 |                 | actually built  |                 |
|                 |                 | upon            |                 |
|                 |                 | scikit-learn    |                 |
|                 |                 | code, even      |                 |
|                 |                 | though it is    |                 |
|                 |                 | implemented in  |                 |
|                 |                 | TensorFlow.     |                 |
|                 |                 | Also, this book |                 |
|                 |                 | deals with the  |                 |
|                 |                 | most famous     |                 |
|                 |                 | machine         |                 |
|                 |                 | learning        |                 |
|                 |                 | problems, like  |                 |
|                 |                 | the MNIST       |                 |
|                 |                 | dataset.        |                 |
|                 |                 | Although this   |                 |
|                 |                 | is not directly |                 |
|                 |                 | related to our  |                 |
|                 |                 | problem, which  |                 |
|                 |                 | is a limitation |                 |
|                 |                 | of the book, it |                 |
|                 |                 | is still really |                 |
|                 |                 | interesting to  |                 |
|                 |                 | observe how     |                 |
|                 |                 | powerful        |                 |
|                 |                 | machine         |                 |
|                 |                 | learning can    |                 |
|                 |                 | actually be if  |                 |
|                 |                 | implemented     |                 |
|                 |                 | correctly.      |                 |
+-----------------+-----------------+-----------------+-----------------+
| 3\. Pearl,      | This book       | This book is    | Can you ever    |
| Judea, and      | offers a        | very relevant   | completely      |
| Dana            | thorough        | to our work     | prove direct    |
| Mackenzie.      | explanation of  | because we need | causation       |
| The Book of     | causal          | to determine if | between any two |
| Why: The New    | inference and   | the datasets we | variables? It   |
| Science of      | the statistical | will use in our | is important to |
| Cause and       | analysis behind | network help    | be very careful |
| Effect.         | cause and       | determine the   | of declaring    |
| Penguin         | effect. Even    | predictions     | things like     |
| Books, 2018.    | more than 20    | because they    | this because of |
|                 | years ago,      | have a causal   | examples        |
|                 | statisticians   | effect on the   | described in    |
|                 | could only      | stock or just   | the book where  |
|                 | prove           | because they    | people have     |
|                 | correlation,    | are correlated  | irrationally    |
|                 | not causation,  | with the stock. | concluded       |
|                 | but with new    | Things like     | things like     |
|                 | scientific      | Google search   | smoking         |
|                 | methods the     | mentions may be | actually        |
|                 | work on         | a direct result | reduced infant  |
|                 | causality is    | of a large      | mortality.      |
|                 | finally being   | change in a     |                 |
|                 | tackled.        | stock's value,  |                 |
|                 |                 | it may be the   |                 |
|                 |                 | cause, or it    |                 |
|                 |                 | may even be     |                 |
|                 |                 | neither and     |                 |
|                 |                 | confounding     |                 |
|                 |                 | variables could |                 |
|                 |                 | have a role.    |                 |
|                 |                 | Understanding   |                 |
|                 |                 | cause and       |                 |
|                 |                 | effect          |                 |
|                 |                 | relationships   |                 |
|                 |                 | is thus         |                 |
|                 |                 | important in    |                 |
|                 |                 | gauging the     |                 |
|                 |                 | usefulness of   |                 |
|                 |                 | the datasets we |                 |
|                 |                 | will use, and   |                 |
|                 |                 | if we should    |                 |
|                 |                 | continue using  |                 |
|                 |                 | them. Obviously |                 |
|                 |                 | this is a book  |                 |
|                 |                 | centered around |                 |
|                 |                 | statistics, so  |                 |
|                 |                 | much of the     |                 |
|                 |                 | actual          |                 |
|                 |                 | equations and   |                 |
|                 |                 | many of the     |                 |
|                 |                 | concepts are    |                 |
|                 |                 | irrelevant to   |                 |
|                 |                 | our work, which |                 |
|                 |                 | is a            |                 |
|                 |                 | limitation.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| 4\. Heinz,      | This is         | This is helpful | Google invests  |
| Sebastian. "A   | essentially a   | and something   | a lot of time   |
| Simple Deep     | tutorial on how | that we could   | and money into  |
| Learning        | to use          | build off of    | AI and ML...    |
| Model for       | tensorflow for  | for our         | How are they    |
| Stock Price     | machine         | algorithm. The  | using it for    |
| Prediction      | learning with   | strategies used | themselves? How |
| Using           | S&P500 data.    | are somewhat    | much profit are |
| TensorFlow."    |                 | generic, but    | they making     |
| Medium, ML      |                 | the author uses | from doing all  |
| Review, 9       |                 | very helpful    | of this         |
| Nov. 2017,      |                 | illustrations   | research? What  |
| medium.com/mlre |                 | for visualizing | makes it worth  |
| view/a-simple-d |                 | the learning    | it for them?    |
| eep-learning-mo |                 | process.        |                 |
| del-for-stock-p |                 | Understanding   |                 |
| rice-prediction |                 | the process is  |                 |
| -using-tensorfl |                 | key to our      |                 |
| ow-30505541d877 |                 | ability to      |                 |
| .               |                 | optimize our    |                 |
|                 |                 | algorithm.      |                 |
+-----------------+-----------------+-----------------+-----------------+
| 5\. "Machine    | This article    | This article    | How helpful     |
| Learning for    | outlines how    | argues that     | will machine    |
| Trading -       | machine         | machine         | learning in 10  |
| Topic           | learning is     | learning is     | years, once it  |
| Overview."      | being used in   | very relevant   | becomes         |
| Sigmoidal,      | the stock       | in trading, and | industry        |
| Sigmoidal       | market and how  | is able to      | standard and    |
| LLC, 15 Oct.    | profitable it   | boost profits.  | every large     |
| 2018,           | has been. It    | This confirms   | firm is even    |
| sigmoidal.io/ma | also briefly    | our original    | again? What     |
| chine-learning- | recognizes      | theories and    | will be the     |
| for-trading/.   | google trends   | helps guide us  | next wave of    |
|                 | as a viable     | in our overall  | profitable      |
|                 | source for      | goal. It        | technology in   |
|                 | trading         | provides some   | the stock       |
|                 | information.    | insight into    | market? Will    |
|                 |                 | our next steps  | machine         |
|                 |                 | once we get a   | learning be the |
|                 |                 | basic algorithm | last large      |
|                 |                 | running. It     | step? What will |
|                 |                 | also highlights | give companies  |
|                 |                 | the importance  | an edge?        |
|                 |                 | of              |                 |
|                 |                 | understanding   |                 |
|                 |                 | machine         |                 |
|                 |                 | learning. It    |                 |
|                 |                 | may not help us |                 |
|                 |                 | with the        |                 |
|                 |                 | technical       |                 |
|                 |                 | details so much |                 |
|                 |                 | as the overall  |                 |
|                 |                 | direction.      |                 |
+-----------------+-----------------+-----------------+-----------------+
| 6\. Milosevic,  | This is an      | This is a       | What algorithms |
| Nikola.         | academic paper  | different       | are optimized   |
| "Equity         | where students  | approach than   | for short term  |
| Forecast:       | created an      | we have looked  | and which are   |
| Predicting      | algorithm for   | at before, and  | for long term?  |
| Long Term       | predicting long | is very         | Which of the    |
| Stock Price     | term stock      | interesting. So | inputs this     |
| Movement        | prices. They    | far, we've been | study uses      |
| Using Machine   | were successful | training on a   | would be        |
| Learning."      | in \~75% of     | year or two of  | helpful for     |
| ArXiv, 2 Mar.   | cases in        | data and        | short term      |
| 2016,           | predicting if a | predicting      | algorithms as   |
| arxiv.org/ftp/a | company would   | \<100 days in   | well as long    |
| rxiv/papers/160 | rise 10% in a   | the future. Our | term? Where did |
| 3/1603.00751.pd | year or not.    | algorithm would | they collect    |
| f.              | They also used  | be optimized    | their data      |
|                 | some other      | for day         | from? Is        |
|                 | qualities of an | trading, where  | trading on long |
|                 | equity's        | this paper is   | term            |
|                 | finances as     | studying years  | predictions     |
|                 | inputs.         | ahead. This is  | viable with     |
|                 |                 | a different     | technology      |
|                 |                 | approach that   | changing the    |
|                 |                 | we could try,   | way markets act |
|                 |                 | but it would    | so rapidly?     |
|                 |                 | require a       |                 |
|                 |                 | different       |                 |
|                 |                 | approach on     |                 |
|                 |                 | training        |                 |
|                 |                 | algorithms.     |                 |
|                 |                 | Regardless, it  |                 |
|                 |                 | is good to be   |                 |
|                 |                 | thinking of the |                 |
|                 |                 | idea of short   |                 |
|                 |                 | term vs long    |                 |
|                 |                 | term stock      |                 |
|                 |                 | predictions,    |                 |
|                 |                 | and will help   |                 |
|                 |                 | us find where   |                 |
|                 |                 | we want to be   |                 |
|                 |                 | on that scale.  |                 |
+-----------------+-----------------+-----------------+-----------------+
| 7. Koehrsen,    | This article is | This article    | How often does  |
| William. "Stock | about someone   | doesn't go into | data            |
| Prediction in   | who uses the    | the technical   | manipulation    |
| Python --       | stocker         | details, but    | occur? In an    |
| Towards Data    | platform for    | does provide a  | almost cynical  |
| Science."       | data on Amazon. | brief story     | sense, the      |
| Towards Data    | They start with | that is helpful | article talks   |
| Science,        | a rudimentary   | for             | about how to    |
| Towards Data    | strategy and    | understanding   | fudge the data  |
| Science, 19     | build it up     | what steps to   | if you obtain   |
| Jan. 2018,      | over time.      | take to move    | undesirable     |
| towardsdatascie |                 | forward in      | results, which  |
| nce.com/stock-p |                 | refining our    | we obviously    |
| rediction-in-py |                 | algorithms. We  | will not do,    |
| thon-b66555171a |                 | can take the    | but how much    |
| 2.              |                 | methods they    | does this lying |
|                 |                 | used into       | occur in the    |
|                 |                 | consideration   | actual field?   |
|                 |                 | when designing  |                 |
|                 |                 | our own         |                 |
|                 |                 | algorithms.     |                 |
|                 |                 | However, the    |                 |
|                 |                 | use a different |                 |
|                 |                 | source for      |                 |
|                 |                 | their stock     |                 |
|                 |                 | data then we do |                 |
|                 |                 | and the syntax  |                 |
|                 |                 | thus differs by |                 |
|                 |                 | a considerable  |                 |
|                 |                 | amount, so it   |                 |
|                 |                 | is more the big |                 |
|                 |                 | ideas that we   |                 |
|                 |                 | are trying to   |                 |
|                 |                 | gain an         |                 |
|                 |                 | understanding   |                 |
|                 |                 | of.             |                 |
+-----------------+-----------------+-----------------+-----------------+
| 8\. Singh,      | This article    | This article is | What are the    |
| Aishwarya.      | describes       | very helpful in | different types |
| "Predicting     | various machine | describing      | of LSTM         |
| the Stock       | learning        | which           | implementations |
| Market Using    | algorithms and  | algorithms      | and how do they |
| Machine         | how effective   | don't work and  | work? How can   |
| Learning and    | they are. It    | why, and leads  | we incorporate  |
| Deep            | works           | to the          | other types of  |
| Learning."      | progressively   | conclusion that | data in with    |
| Analytics       | toward more     | the Long Short  | the LSTM        |
| Vidhya, 26      | complex         | Term Memory     | prediction?     |
| Oct. 2018,      | algorithms and  | (LSTM)          |                 |
| www.analyticsvi | explain the     | algorithm might |                 |
| dhya.com/blog/2 | concepts behind | lead to the     |                 |
| 018/10/predicti | each algorithm. | best results.   |                 |
| ng-stock-price- |                 | It doesn't go   |                 |
| machine-learnin |                 | in detail into  |                 |
| gnd-deep-learni |                 | how LSTM works, |                 |
| ng-techniques-p |                 | it does give a  |                 |
| ython/.         |                 | brief overview  |                 |
|                 |                 | and             |                 |
|                 |                 | demonstrates    |                 |
|                 |                 | how it can be   |                 |
|                 |                 | implemented. It |                 |
|                 |                 | also explains   |                 |
|                 |                 | algorithms that |                 |
|                 |                 | aren't so       |                 |
|                 |                 | accurate, but   |                 |
|                 |                 | help build an   |                 |
|                 |                 | understanding   |                 |
|                 |                 | of how LSTM     |                 |
|                 |                 | works.          |                 |
+-----------------+-----------------+-----------------+-----------------+
| 9\. Braun,      | A program that  | This is a very  | How do          |
| Max.            | uses references | helpful example | presidential    |
| "Trump2Cash."   | to stocks in    | of how we can   | tweets differ   |
| GitHub, 22      | the president's | use twitter     | from those of   |
| Sept. 2018,     | tweets to       | feeds to        | economic        |
| github.com/maxb | predict stocks  | provide insight | advisors and    |
| braun/trump2cas | in python       | on stock        | popular stock   |
| h.              |                 | prices. It      | brokers in      |
|                 |                 | helps reinforce | their           |
|                 |                 | our original    | influence? What |
|                 |                 | hypotheses on   | other popular   |
|                 |                 | what affects    | social media    |
|                 |                 | stocks and      | platforms can   |
|                 |                 | gives us        | be analyzed?    |
|                 |                 | potential tools |                 |
|                 |                 | to use in       |                 |
|                 |                 | python. It      |                 |
|                 |                 | isn't in and of |                 |
|                 |                 | itself a tool   |                 |
|                 |                 | that we can     |                 |
|                 |                 | use, but a      |                 |
|                 |                 | reference for   |                 |
|                 |                 | how to complete |                 |
|                 |                 | some specific   |                 |
|                 |                 | tasks.          |                 |
+-----------------+-----------------+-----------------+-----------------+
| 10\. Hilpisch,  | This is an      | This is very    | What do the     |
| Yves.           | article about a | helpful because | professionals   |
| "Algorithmic    | script that     | it shows a      | use? What are   |
| Trading in      | uses time       | different       | the best        |
| Less than 100   | series momentum | platform that   | platforms that  |
| Lines of        | strategy and    | we can use to   | provide         |
| Python Code."   | the platform    | gather data.    | detailed data   |
| O\'Reilly       | Oanda to        | Although it     | quickly and in  |
| Media, 18       | predict on      | isn't an in     | an easy to use  |
| Jan. 2017,      | backlogs of     | depth           | format? How     |
| www.oreilly.com | data and        | explanation of  | effective will  |
| /learning/algor | compare         | every step,     | the training on |
| ithmic-trading- | different       | this is a       | one stock be on |
| in-less-than-10 | variations of   | script that we  | another stock?  |
| 0-lines-of-pyth | the strategies  | could work off  |                 |
| on-code.        | to maximize     | of to improve   |                 |
|                 | potential       | with other      |                 |
|                 | profits.        | inputs. It also |                 |
|                 |                 | mentions        |                 |
|                 |                 | Quantopian,     |                 |
|                 |                 | which is an     |                 |
|                 |                 | online tool for |                 |
|                 |                 | writing and     |                 |
|                 |                 | testing         |                 |
|                 |                 | algorithms,     |                 |
|                 |                 | which could     |                 |
|                 |                 | also be useful. |                 |
+-----------------+-----------------+-----------------+-----------------+

\newpage

# Thank You Letters to Community Members

## To Eddie Kohler, Harvard CompSci 61 Professor
> Hi Eddie,
> 
> Thanks so much for the class! It was a fantastic experience and I learned so much. I really appreciate your generosity in going out of your way to let high school students to take your class and taking on the extra work that comes with it. On top of that you were a great teacher and made the class very interesting and a lot of fun. A number of CS interested students have approached me inquiring about classes at Harvard, and so if you are still willing next year, there's a good chance that a few of them will be interested.
> 
> Happy New Years!
> 
> Best,
> 
> Max Katz-Christy and David Inho Lee

\newpage

# Formative Reflections

- Logistic Regression Model: We were able to build an effective logistic regression machine learnign model fairy easily. The hardest task was figuring out the best way to obtain the data that we needed. We spent a lot of time looking for the best API for getting consistant, reliable stock data. After deciding on the API, pandas-datareader, we spent time playing around with the library to see what data we could access. We ended up finding out we could reliably get daily stock data for the past five years, giving us approximately 1250 points of data. We then theorized about early preprocessing methods, and organized the stock data into the format we wanted. Creating the actual machine learning model was pretty simple after that, especially because we had a lot of experience with this type of model from previous internships.

![](demos/logistic_reg_model.png?raw=true)

- Early Neural Network Development: We had many challenges when first implementing a neural network for our project. It was challenging to figure out how to shape the data to the very specific requirements of the network. Even after finishing this, we had to figure out what to do with all the hyperparameters of the actual network. After completing the very first generation of our network, the performance we were getting was terrible. The model performed fairly well for the first few days of the stock data, but then quickly predicted linearly after that. 
(insert image from slide 17 here)
The most amount of time spent during this whole process was trying to figure out how exactly we needed to normalize our data so that the neural network would put the same amount of emphasis to each day of data instead of the first few days like we were thinking. We also had to keep tinkering with the number of epochs, the size of each layer, the number of layers, whether or not to include dropout or other types of layers, loss function, etc. to find the best possible performance for our network. But after working at this problem for a long time, we eventually landed a model that we were happy with and was performing decently well.

![](demos/early_neural_network_devel.png?raw=true)

- Later Neural Network Development: We decided as a next step to incorporate more data into the network, as the ~1250 samples we were using for our input was not very large at all compared to the hundreds of thousands to millions of data samples that is usually recommended for machine learning models. We thus ended up looking for an API that could access twitter data or some other data relevant to internet mentions. We landed on using the pytrends package which accesses Google Trends data. This worked well for us because it would give the network information on how much the stock of interest was being mentioned in the news, which definitely has a direct effect on the actual stock price. We had to restructure our entire data input to accommodate for this new source of data, but after finally implementing this we saw incredible results. Our network’s prediction now precisely followed the stock data without showing many signs of overfitting. The loss of the network was at its lowest point yet and future stock predictions appeared legitimate. At this point, we could start to finalize the structure of our model.

![](demos/later_neural_network_devel.png?raw=true)

- Final Steps and Testing: As a next step, we needed to make our network actually usable to outside users. We decided to create a script using the stockbot class we had created to predict what the best and worst performing stocks would be in the future. The actual code implementation of this idea was not hard at all, we just needed to decide on the specifics of the numbers we would use. We ended up deciding that a one month testing period was a good enough heuristic - our model would take the top 29 stocks in the world, train and predict on the data for each of them, and predict how each of them would perform in the next month, saving multiple graphs to display the predicted performance of these stocks. We also created a checking script to evaluate the performance of our model. We found that on average our network would predict whether a stock would rise or fall in the next month correctly for 28 out of the 29 stocks. The only difficult part about this final process was that the running time for our code at this point neared two hours, so if there was any bug in our code it took a long time to actually be able to fix.

![](demos/results_all.png?raw=true)
![](demos/results_top.png?raw=true)

\newpage

# Summative Reflection Letter

Dear Presentation Panel,

Working on our graduation project this semester was a very unique and fulfilling experience for the both of us. We both had done work in the past either directly or indirectly relating to the field of machine learning, but this was the first time we were given such independence to do whatever we wanted. We landed on the idea of creating a stock bot because it was an idea we had vaguely heard about but didn’t know too much about and it was something very different from what either of us had done in the past. The methodology of our work process was a little different as well. We were used to being told where to start and what resources to look at to begin, but in this project we had to come up with everything from scratch. For us this was a huge plus, the process of brainstorming, which both of us had been doing since the previous year, made our project in the end really feel like it was ours and made the whole experience a lot more fulfilling.

Looking back on our project, we are really happy about what we were able to accomplish and where we ended up finishing. We both were honestly astounded by how well our model was able to perform in the end: we had an average yearly return rate 14 times greater than that of the overall stock market, which we thought was more than impressive in any context. We successfully were able to incorporate more than just stock data into our model, which was another one of our goals, and the overall user friendliness and object orientedness of our code was very satisfying. There are however a couple of things we would have liked to add to our project that we did not get to. The final two scripts we ended up writing to evaluate model performance and give future stock predictions are honestly pretty limited in scope, so being able to make these into full fledged libraries in the future is a direction we could head in. We also could always add more data sources, whether that be twitter mentions or news reports or something we haven’t even thought about yet, and we never got to the point where we could use smaller neural networks to predict the data to go into the larger neural networks.

Comparing our work to the literature that is out there, we have some similarities and differences. The overall structure of our data and our network is largely similar to those who have done similar work to make stockbots like ours: the major difference is the methodology in which we came to the final product. We made it a point to ourselves to write our entire project completely ourselves: obviously we could consult stack overflow and other sources for the more technical aspects of our code, but we did not want to just reuse code that someone had already written and just get the same results as them. Because of this choice, our process probably took a lot longer than it should have. We are both amateurs in this field, so there were a lot of things we did not think about in the early stages of our project that came around to bite us in the back later on, costing us a lot of time that we could have used to add more functionality. But we think it is because of our painstaking methodology that we have been able to gain so much more knowledge about what we did as well as machine learning as a broader field.

One thing we would like to do in the future is to create a fully fledged website, application, or tool using the network and related scripts we created. Making our work user friendly aesthetically pleasing to look at should be a high priority especially given the complex nature of our project. Having an easily accessible place to go would be a really nice way to be able to show off all our work to somebody who might not be that tech savy or just in a quick and easy way. We also never got to actually test our model on the real world by investing actual money into the stock market. We think it would be really cool and interesting to invest a small amount of money into the stock market based on what our model says we should do, and evaluate how well it did by the end of second semester, kind of as a followup to this project. It would really show that we have faith in what we created and would be a great ending to this journey that has had all of its ups and downs. But overall, we are really happy with what we were able to accomplish and we will both definitely continue to pursue this type of work in the future.

Sincerely,
David and Max
