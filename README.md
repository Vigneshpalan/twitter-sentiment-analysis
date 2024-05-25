

# Twitter Sentiment Analysis Using Logistic Regression

This project implements a sentiment analysis model for Twitter data using a logistic regression algorithm. The steps involved in the project are:

1. **Preprocessing of Twitter data**
2. **Extracting features**
3. **Training a logistic regression model from scratch**
4. **Testing the logistic regression model**

## Table of Contents

- [Preprocessing of Twitter Data](#preprocessing-of-twitter-data)
- [Extracting Features](#extracting-features)
- [Training Logistic Regression Model](#training-logistic-regression-model)
- [Testing Logistic Regression Model](#testing-logistic-regression-model)
- [Results](#results)
- [Conclusion](#conclusion)

## Preprocessing of Twitter Data

The preprocessing steps include:

1. **Removing hyperlinks, Twitter marks, and styles:** 
   - Clean the text data by removing URLs, Twitter handles, and any special styles that might be present.
   
2. **Tokenizing and Lowercasing:** 
   - Split the text into individual tokens (words) and convert them to lowercase for uniformity.
   
3. **Removing Stopwords and Punctuation:** 
   - Remove common stopwords (e.g., "and", "the", "is") and punctuation to reduce noise in the data.

## Extracting Features

We make use of the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text data into a numeric matrix representation. This helps in quantifying the importance of words in the dataset.

## Training Logistic Regression Model

The logistic regression model is trained from scratch using a gradient descent function to minimize the cost function. The steps include:

1. **Initializing Weights:**
   - Randomly initialize the weight vector.

2. **Computing the Cost Function:**
   - Use the logistic loss function to compute the cost.


The logistic regression model is extended to solve the multi-class classification problem with three classes: 
- 0: Negative
- 2: Neutral
- 4: Positive

## Testing Logistic Regression Model

The trained model is used to predict the sentiment of the test data. The model's accuracy is then evaluated based on these predictions.

## Results

The model achieves an accuracy of 87% on the test data. This indicates a strong performance in classifying the sentiments of the tweets.

## Conclusion

This project demonstrates the implementation of a sentiment analysis model using logistic regression. The preprocessing steps, feature extraction using TF-IDF, and training the logistic regression model. The model's accuracy of 87% shows its effectiveness in sentiment analysis tasks for Twitter data.

---


### Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- NLTK

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```
   cd twitter-sentiment-analysis
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

branches or files.
