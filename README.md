
# Fake and Real News Detection Using Machine Learning

## Project Overview

This project develops an NLP-based Machine Learning system to classify news articles as Fake News or Real News.

The system uses Natural Language Processing, TF-IDF feature extraction, multiple Machine Learning algorithms, model evaluation, probability calibration, and a Streamlit web application.

**Important:** This is an experimental text classifier. It identifies patterns learned from a historical dataset; it does not independently verify factual accuracy.

## Dataset

Dataset: Fake and Real News Dataset by Clément Bisaillon (Kaggle).

Dataset link:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Files:
- Fake.csv
- True.csv

Original dataset:
- Fake articles: 23,481
- Real articles: 21,417
- Total: 44,898

After preprocessing and removing empty articles:
- Fake articles: 22,763
- Real articles: 21,210
- Total: 43,973

Label mapping:
- 0 = Fake News
- 1 = Real News

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Workflow

### 1. Data Collection and Exploration
- Loaded Fake.csv and True.csv.
- Examined dataset dimensions, columns, data types and missing values.
- Analysed class distribution and duplicate records.

### 2. Text Preprocessing
- Lowercase conversion.
- URL and HTML removal.
- Number and punctuation removal.
- Tokenization.
- Stopword removal while preserving negation words.
- Porter stemming.
- Removal of empty cleaned articles.

### 3. Feature Engineering

Used TF-IDF (Term Frequency–Inverse Document Frequency).

Configuration:
- Maximum features: 20,000
- N-gram range: (1, 2)
- Minimum document frequency: 2
- Maximum document frequency: 0.95
- Sublinear term frequency: True

### 4. Machine Learning Models

Six classification algorithms were trained and compared:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Passive Aggressive Classifier
4. Decision Tree
5. Random Forest
6. Linear Support Vector Machine (SVM)

### 5. Model Evaluation

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Average Precision

The experiments included stratified cross-validation, hyperparameter tuning, and group-based and temporal validation.

### 6. Data Leakage and Robustness Analysis

Investigated:
- Repeated article titles across random splits.
- Group-based splitting to prevent title overlap.
- Reuters source-marker dependence.
- Performance after removing Reuters markers.
- Title-only, body-only and combined-text experiments.
- Temporal validation using article publication dates.
- Duplicate-title effects on evaluation.

### 7. Explainability and Calibration

- Analysed Linear SVM feature coefficients.
- Examined individual prediction contributions.
- Calibrated model probabilities using sigmoid calibration.
- Evaluated Brier score and Log Loss.
- Analysed prediction confidence.
- Implemented selective classification with a 90% confidence threshold.

### 8. Streamlit Application

The web application provides:
- News article text input.
- Fake and Real probability estimates.
- Model confidence score.
- Manual-review recommendation for low-confidence predictions.
- Prediction history.
- Clear prediction history functionality.

## Important Experimental Results

### Group-based SVM

- Accuracy: 99.45%
- Precision: 99.17%
- Recall: 99.69%
- F1-score: 99.43%

### Temporal SVM without Reuters markers

- Accuracy: 98.97%
- Precision: 99.64%
- Recall: 99.14%
- F1-score: 99.39%

These results are specific to the dataset and evaluation splits used. They should not be interpreted as evidence of equivalent performance on current real-world news.

## Project Structure

```text
Fake-and-Real-News-Detection/
│
├── models/
│   └── fake_news_detector.joblib
│
├── app.py
├── Fake News Detector.ipynb
├── Fake.csv
├── True.csv
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Fake-and-Real-News-Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Model File

The trained model is stored at:

`models/fake_news_detector.joblib`

The saved package contains:
- Calibrated classification model.
- Confidence threshold.
- Label mapping.

## Limitations

- The dataset contains historical articles primarily from 2015–2018.
- Dataset-specific writing and publisher patterns may influence predictions.
- High model confidence does not guarantee factual correctness.
- The model is not connected to external fact-checking services.
- Independent external validation is needed before real-world use.

## Future Improvements

- Evaluate on newer and independent news datasets.
- Test generalisation across publishers and topics.
- Explore transformer-based NLP models such as BERT.
- Add source credibility analysis.
- Improve model monitoring and explainability.
- Deploy the application publicly.

## Author

Developed as an end-to-end Machine Learning and NLP project.
