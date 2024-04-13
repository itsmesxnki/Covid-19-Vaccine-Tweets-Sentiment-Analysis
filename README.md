## Sentiment Analysis of Twitter Data

This repository contains Python scripts for performing sentiment analysis on Twitter data using machine learning classifiers. The sentiment analysis aims to classify tweets into positive, negative, or neutral sentiments.

### Code Overview

1. **Preprocessing and Naive Bayes Classifier (`preprocessing_nb_classifier.py`):**
   - This script preprocesses the Twitter data, including cleaning and tokenization.
   - It then trains a Naive Bayes classifier using the preprocessed data.
   - Finally, it evaluates the classifier's performance using confusion matrix and classification report.

2. **Preprocessing and Support Vector Machine Classifier (`preprocessing_svm_classifier.py`):**
   - Similar to the first script, this one preprocesses the Twitter data.
   - It then trains a Support Vector Machine classifier using the preprocessed data.
   - The script evaluates the classifier's performance using confusion matrix and classification report.

3. **Preprocessing and Logistic Regression Classifier (`preprocessing_lr_classifier.py`):**
   - This script preprocesses the Twitter data.
   - It then trains a Logistic Regression classifier using the preprocessed data.
   - The script evaluates the classifier's performance using confusion matrix and classification report.

4. **Advanced Preprocessing and Classifier Comparison (`advanced_preprocessing_and_comparison.py`):**
   - This script implements more advanced text preprocessing techniques, including URL and hashtag removal.
   - It trains multiple classifiers (Naive Bayes, SVM, and Logistic Regression) using the preprocessed data.
   - Finally, it compares the performance of these classifiers and visualizes the results using a bar chart.

### Getting Started

1. **Clone the Repository:**
   ```
   git clone https://github.com/your_username/sentiment-analysis-twitter.git
   ```

2. **Install Dependencies:**
   - Ensure you have Python installed on your machine.
   - Install the required Python packages by running:
     ```
     pip install -r requirements.txt
     ```

3. **Download the Dataset:**
   - Download the Twitter dataset (`tweets_data.csv`) and place it in the project directory.

4. **Run the Scripts:**
   - Execute any of the provided Python scripts based on your preference:
     ```
     1.1.py
     1.py
     2.1.py
     2.py
     ```

### Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or open an issue.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
