

# 1. Spam Detection Using Machine Learning

## Project Overview

This project demonstrates how to build a **Spam Detection** system using machine learning techniques in Python. The goal is to classify SMS messages or emails as **spam** (unwanted, unsolicited messages) or **ham** (legitimate messages) by preprocessing the text data, extracting meaningful features, training classifiers, and evaluating their performance.

The project is implemented in a Jupyter Notebook for easy experimentation and visualization.

## Features

- **Data Preprocessing:** Clean and prepare raw text data for analysis.
- **Feature Extraction:** Use TF-IDF vectorization to convert text into numerical features.
- **Model Training:** Train classification models including:
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)
- **Evaluation:** Measure model accuracy, precision, recall, and F1-score.
- **Visualization:** Plot pie charts to show the distribution of spam vs ham messages.
- **Prediction:** Classify new, unseen messages as spam or ham.

## Dataset

The project uses the **Email Spam Collection Dataset**, a public dataset containing 5,574 SMS messages labeled as spam or ham.

- Dataset source: [UCI Machine Learning Repository](https://github.com/Santanu-hub25/OutriX_tasks/blob/main/spam.csv)
- Alternatively, the dataset is loaded directly from a GitHub raw URL in the notebook.

## Technologies & Libraries

- Python 3.11.3
- Jupyter Notebook
- pandas
- scikit-learn
- matplotlib

## How to Run

1. Clone this repository:

GitHub clone https://github.com/Santanu-hub25/OutriX_tasks/tree/main

2. Open the Jupyter Notebook:
http://localhost:8888/notebooks/Downloads/jupyter%20projects/Spam%20Detection%20Using%20ML.ipynb

3. Install required Python packages (preferably in a virtual environment):
!pip install numpy pandas scikit-learn matplotlib


4. Run the notebook cells sequentially to:
- Load and preprocess data
- Extract TF-IDF features
- Train Naive Bayes and SVM classifiers
- Evaluate model performance
- Visualize data distribution
- Test predictions on new messages


## Results

- The models achieve **~98-99% accuracy** on the test set.
- SVM slightly outperforms Naive Bayes in precision and recall.
- The pie chart visualization clearly shows the class imbalance between ham and spam messages.


---

# 2. Live Twitter Sentiment Analysis Pipeline

This project implements a complete pipeline to perform **sentiment analysis** on live tweets. It collects tweets in real-time from the Twitter API, preprocesses the tweet texts, classifies their sentiment labels (Positive, Negative, Neutral), and visualizes sentiment trends.




##  Project Overview

Social media platforms like Twitter provide real-time public opinion. This project demonstrates how to capture live tweets, clean noisy text data, and apply sentiment analysis to understand public mood about specific topics.

The sentiment classification uses **TextBlob** for polarity scoring to label tweets as Positive, Negative, or Neutral, and visualizations include pie charts, bar charts, and word clouds.


## Features

- Collect live tweets using Twitter API with Tweepy
- Text preprocessing: cleaning, tokenization, stopword removal, and lemmatization
- Sentiment classification (positive/negative/neutral) using TextBlob polarity
- Sentiment distribution visualization (pie chart, bar plot)
- Word cloud visualization per sentiment class


## Installation

1. Clone the repository:
Git clone https://github.com/Santanu-hub25/OutriX_tasks/blob/main/Sentiment%20Analysis%20of%20Twitter%20Data.ipynb


2. Install required libraries:
!pip install tweepy textblob matplotlib pandas nltk wordcloud


3. Download NLTK data (run inside Jupyter or Python environment):
- import nltk
- nltk.download('punkt')
- nltk.download('wordnet')
- nltk.download('stopwords')


## Usage

1. Obtain Twitter API credentials from [Twitter Developer Portal](https://developer.twitter.com/).

2. Open the Jupyter Notebook `twitter_sentiment_analysis.ipynb`.

3. Replace API keys in the authentication cell with your credentials:

- API_KEY = 'YOUR_API_KEY'  
- API_SECRET = 'YOUR_API_SECRET'
- ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
- ACCESS_TOKEN_SECRET = 'YOUR_ACCESS_TOKEN_SECRET'


4. Run cells sequentially to:

- Collect live tweets by keyword
- Preprocess tweets
- Perform sentiment classification
- Visualize sentiment trends and word clouds

## Dependencies

- Python 3.x
- tweepy
- pandas
- nltk
- textblob
- matplotlib
- seaborn
- wordcloud

## Results

After running the sentiment analysis pipeline on 100 recent tweets containing the keyword **"vaccine"**, the following results were observed:

### Sentiment Distribution

The classifier categorized the tweets into three sentiment classes:

| Sentiment | Number of Tweets | Percentage |
|-----------|------------------|------------|
| Positive  | 43               | 43%        |
| Neutral   | 36               | 36%        |
| Negative  | 21               | 21%        |

*Note: Numbers can vary depending on the live data fetched.*

### Visualizations

1. **Pie Chart**  
Shows the proportion of positive, neutral, and negative tweets, helping to quickly assess the overall public sentiment.


2. **Bar Chart**  
Depicts tweet counts by sentiment category.

3. **Word Clouds**  
Visualizations show the most frequent words in tweets per sentiment class, highlighting common topics in positive, negative, and neutral discussions.


### Insights

- The majority of tweets were **positive**, indicating general optimism or support related to vaccines.
- Neutral tweets mainly involved information sharing or announcements.
- Negative tweets often contained concerns or complaints, which could help identify public issues or misinformation.

## Acknowledgements

- [Tweepy](https://www.tweepy.org/) - Twitter API client
- [TextBlob](https://textblob.readthedocs.io/en/dev/) - Text processing and sentiment analysis
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit
- Inspiration from Real Python tutorials and open-source sentiment analysis projects.


## Contact

For questions or suggestions, please open an issue or contact [santanuworkspace25@gmail.com].

Thank you for checking out this projects !



