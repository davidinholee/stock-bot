Thesis and Graduation Project

Literature Review --Notes

[]{#t.530217ad236de837b6bf3c5b8052983342546846}[]{#t.0}

+-----------------+-----------------+-----------------+-----------------+
| Use EasyBib to  | Brief Summary   | Assess the uses | New Questions   |
| create          |                 | and limits of   |                 |
| bibliographic   |                 | this work in    |                 |
| citations in    |                 | relation to     |                 |
| the citation    |                 | your own        |                 |
| style of the    |                 | thinking        |                 |
| discipline      |                 |                 |                 |
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
