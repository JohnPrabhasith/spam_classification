# Spam Classification Project

## Overview
This project demonstrates a simple implementation of a spam classification system using machine learning. It employs Bag-of-Words (BoW) and TF-IDF vectorization techniques to preprocess text data and trains a classifier to identify spam messages.

---

## Features
- Preprocesses raw text data for analysis.
- Vectorizes text using Bag-of-Words and TF-IDF.
- Trains and evaluates a machine learning classifier (e.g., Random Forest or Naive Bayes).
- Includes performance metrics such as accuracy, precision, recall, and F1-score.

---

## Prerequisites
Ensure you have the following installed:

- Python 3.7+
- NumPy
- pandas
- scikit-learn

To install the required libraries, run:

```bash
pip install numpy pandas scikit-learn 
```

---

## Dataset
The project uses the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). The dataset contains labeled SMS messages, with each message categorized as either "ham" (non-spam) or "spam."

### 1. Data Preprocessing
- Load the dataset.
- Remove stop words, punctuations, and convert text to lowercase.
- Tokenize text and apply stemming/lemmatization (optional).

### 2. Feature Extraction
- **Bag-of-Words (BoW):** Count the occurrence of each word in the corpus.
- **TF-IDF:** Transform text into feature vectors based on term frequency and inverse document frequency.

### 3. Model Training
- Train a machine learning model (e.g., Random Forest or Naive Bayes) using the extracted features.

### 4. Model Evaluation
- Evaluate the classifier using metrics such as accuracy_score and Classification report.

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/JohnPrabhasith/spam_classification.git
cd spam_classification
```

### 2. Run the Project
#### Jupyter Notebook:
1. Open the `SpamHamClassification.ipynb` in a Jupyter environment.
2. Follow the steps in the notebook to preprocess the data, train the model, and evaluate performance.

