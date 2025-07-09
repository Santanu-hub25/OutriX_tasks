

# Spam Detection Using Machine Learning

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


## Contact

For questions or suggestions, please open an issue or contact [santanuworkspace25@gmail.com].

Thank you for checking out this spam detection project!
