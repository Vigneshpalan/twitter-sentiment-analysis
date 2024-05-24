Twitter Sentiment Analysis Using Logistic Regression
This project is a implementation of a sentiment analysis model for Teitter data using a logistic regression algorithm. The project includes the following steps:

Preprocessing of Teitter data
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

Extracting Features
The frequency dictionary is used to represent features in numerical form from the text data. A feature for a tweet is a 1x3 vector, where the first element is 1 representing the bias, the second element is the sum of positive frequencies for every word in the tweet that is found in the frequency dictionary as (word,1) pair, and the third element is the sum of negative frequencies for every word in the tweet that appears in the frequency dictionary as (word,0) pair.

Training Logistic Regression Model
A logistic regression model is trained from scratch using a gradient descent function to minimize the cost of training. The final weight vector is used to make predictions.

Testing Logistic Regression Model
The trained model is used to predict the sentiment of test data. If the prediction value is greater than 0.5, the sentiment is labeled as positive sentiment, and if the prediction value is less than 0.5, the sentiment is labeled as negative sentiment.
The accuracy of the model is determined by calculating the number of tweets that have been classified correctly as compared to the total number of tweets.

The model is 87% accurate.
