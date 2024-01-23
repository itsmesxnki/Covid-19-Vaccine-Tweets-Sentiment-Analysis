import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline
import re

# IMPORTING DATASET
try:
    data = pd.read_csv('tweets_data.csv', encoding='utf-8')
except:
    data = pd.read_csv('tweets_data.csv', encoding='ISO-8859-1')

# DATA CLEANING AND PREPROCESSING
data = data.dropna(subset=['text', 'label']).astype(str)

# Custom text preprocessing function to handle noise
def preprocess_text(text):
    # Remove URLs, user mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text, flags=re.MULTILINE)

    # Tokenize using a specialized TweetTokenizer for better handling of Twitter-specific text
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    return ' '.join(tokens)

# EXTRACTING INDEPENDENT AND DEPENDENT VARIABLES
n = 35000
x = data['text'].head(n)
y = data['label'].head(n)

# SPLITTING THE DATASET INTO TRAINING AND TESTING SET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FEATURE EXTRACTION AND MODELING WITH PIPELINES
tfidf_vectorizer = TfidfVectorizer(max_features=50000)
count_vectorizer = CountVectorizer(max_features=50000)

nb_classifier = MultinomialNB()
svm_classifier = SVC()
lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)

# Create pipelines with text preprocessing, vectorization, and classifier
nb_pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', tfidf_vectorizer),
    ('classifier', nb_classifier)
])

svm_pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', count_vectorizer),
    ('classifier', svm_classifier)
])

lr_pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', tfidf_vectorizer),
    ('classifier', lr_classifier)
])

# Train and evaluate the models
pipelines = [nb_pipeline, svm_pipeline, lr_pipeline]
classifier_names = ['Naive Bayes', 'SVM', 'Logistic Regression']

for i, pipeline in enumerate(pipelines):
    print(f"Training and evaluating {classifier_names[i]}")
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {classifier_names[i]}:")
    print(cm)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy for {classifier_names[i]}:", accuracy)
    print(report)

# Visualize the accuracy using a bar chart
plt.figure(figsize=(10, 5))
plt.bar(classifier_names, accuracies)
plt.title('Classifier Accuracy')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.show()