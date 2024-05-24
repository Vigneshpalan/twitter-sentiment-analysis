Twitter Sentiment Analysis Using Logistic Regression
This project is a implementation of a sentiment analysis model for Teitter data using a logistic regression algorithm. The project includes the following steps:

Preprocessing of Twitter data
Building frequency dictionary
Extracting features
Training logistic regression model from scratch
Testing logistic regression model

Removing hyperlinks, twitter marks and styles
Tokenizing and lowercasing
Removing stopwords and punctuation
Stemming
Building Frequency Dictionary
A frequency dictionary is built using the tweets and labels in the training dataset. The frequency dictionary consists of unique words in the corpus of tweets after preprocessing and their corresponding positive and negative frequencies.

Training Logistic Regression Model
A logistic regression model is trained from scratch using a gradient descent function to minimize the cost of training. The final weight vector is used to make predictions.

Testing Logistic Regression Model
The trained model is used to predict the sentiment of test data. .

The model is 87% accurate.
