# 📰 Fake and Real News Detection Using Machine Learning

An end-to-end Machine Learning and Natural Language Processing (NLP) project that classifies news articles as **Fake News** or **Real News** using text classification techniques.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Project-Completed-success.svg)]()

---

## 📌 Project Overview

The objective of this project is to develop a Machine Learning-based text classification system capable of identifying patterns associated with fake and real news articles.

The project follows an end-to-end Machine Learning workflow, covering data exploration, text preprocessing, feature engineering, model training, evaluation, explainability, probability calibration and application deployment.

The final application allows users to submit a news article and receive:

- Fake News and Real News probability estimates.
- Predicted classification.
- Model confidence score.
- Manual-review recommendation for low-confidence predictions.
- Prediction history during the current session.
- Downloadable prediction history in CSV format.

**Important:** This is an experimental text classifier. It identifies textual patterns learned from a historical dataset. It does not independently verify factual accuracy, authenticate sources or establish whether a news article is genuine.

---

## 📂 Dataset

This project uses the **Fake and Real News Dataset** created by Clément Bisaillon.

**Dataset source:** [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Dataset files

| File | Description | Records |
|---|---|---:|
| Fake.csv | Fake news articles | 23,481 |
| True.csv | Real news articles | 21,417 |
| **Total** | | **44,898** |

### Dataset columns

- `title`: News article headline.
- `text`: Full article body.
- `subject`: News category.
- `date`: Article publication date.

### Data preparation

The following operations were performed:

1. Assigned binary labels to fake and real news records.
2. Combined both datasets.
3. Randomly shuffled the combined dataset.
4. Removed duplicate records.
5. Cleaned and preprocessed article text.
6. Removed records with empty cleaned text.

### Final dataset distribution

| Class | Label | Records |
|---|---:|---:|
| Fake News | 0 | 22,763 |
| Real News | 1 | 21,210 |
| **Total** | | **43,973** |

The final dataset was used for feature engineering and model development.

---

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine Learning algorithms |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Streamlit | Web application development |
| Git and GitHub | Version control and project hosting |

---

## 🔄 Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand the dataset structure and quality.

Activities included:

- Dataset dimensions and column inspection.
- Data type identification.
- Missing-value analysis.
- Duplicate-record analysis.
- Class distribution analysis.
- Descriptive statistical analysis.

### 2. Text Preprocessing

Used NLTK and Python libraries to clean and normalize article text.

Preprocessing operations:

1. Lowercase conversion.
2. URL removal.
3. HTML tag removal.
4. Number removal.
5. Punctuation removal.
6. Whitespace normalization.
7. Tokenization.
8. Stopword removal.
9. Preservation of negation words such as `no`, `nor`, `not` and `never`.
10. Porter stemming.

The preprocessing pipeline was applied consistently during training and application inference.

### 3. Feature Engineering – TF-IDF

Used **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert cleaned text into numerical feature vectors.

Configuration:

| Parameter | Value |
|---|---|
| max_features | 20,000 |
| ngram_range | (1, 2) |
| min_df | 2 |
| max_df | 0.95 |
| sublinear_tf | True |

Both unigrams and bigrams were included to capture individual words and word combinations.

The TF-IDF vectorizer was fitted on training data only to prevent feature-extraction leakage.

### 4. Machine Learning Model Development

Trained and compared six supervised classification algorithms:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Passive Aggressive Classifier
4. Decision Tree Classifier
5. Random Forest Classifier
6. Linear Support Vector Machine (SVM)

### 5. Model Evaluation

Evaluated the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Average Precision
- Stratified Cross-Validation

The evaluation process also included hyperparameter tuning using GridSearchCV.

### 6. Data Leakage and Robustness Investigation

Investigated whether repeated articles and publisher-specific patterns influenced model performance.

Experiments included:

- Repeated-title analysis.
- Train-test title overlap investigation.
- Group-based train-test splitting.
- Reuters source-marker analysis.
- Model evaluation after removing Reuters markers.
- Title-only classification.
- Article-body-only classification.
- Combined title and article-body classification.
- Temporal train-test splitting.
- Unique-title evaluation.

These experiments helped identify dataset-specific shortcuts and assess how evaluation performance changed under different splitting strategies.

### 7. Explainable AI

Explored model explainability using Linear SVM coefficients.

Implemented:

- Global feature coefficient analysis.
- Identification of features associated with Fake News predictions.
- Identification of features associated with Real News predictions.
- Individual prediction contribution analysis.

Feature contributions were used to understand which words and word combinations influenced individual predictions.

These explanations represent learned dataset associations, not proof of factual accuracy.

### 8. Probability Calibration

Applied sigmoid probability calibration using `CalibratedClassifierCV`.

Evaluated probability quality using:

- Brier Score
- Log Loss
- Reliability Diagram

Calibration was used to convert classifier decision scores into probability estimates.

### 9. Confidence Analysis and Selective Classification

Implemented a confidence-based classification mechanism.

The application uses a **90% confidence threshold**:

| Confidence | Application decision |
|---|---|
| 90% or above | Automatic classification |
| Below 90% | Manual Review Recommended |

The threshold was selected using validation results.

Validation results at the 90% threshold:

- Coverage: 96.10%
- Selective accuracy: 99.82%

These values describe the validation experiment and should not be interpreted as guaranteed real-world performance.

---

## 📊 Model Performance

### 1. Initial Random Train-Test Split

The following results were obtained from the initial stratified random split.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Linear SVM | 99.76% | 99.69% | 99.81% | 99.75% |
| Passive Aggressive | 99.76% | 99.76% | 99.74% | 99.75% |
| Random Forest | 99.65% | 99.72% | 99.55% | 99.63% |
| Decision Tree | 99.52% | 99.69% | 99.32% | 99.50% |
| Logistic Regression | 99.23% | 99.04% | 99.36% | 99.20% |
| Multinomial Naive Bayes | 95.65% | 95.39% | 95.59% | 95.49% |

**Best-performing initial models:** Linear SVM and Passive Aggressive Classifier.

### 2. Cross-Validation Results

Five-fold Stratified Cross-Validation was performed.

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 99.63% | 99.53% | 99.70% | 99.61% |
| Passive Aggressive | 99.62% | 99.53% | 99.68% | 99.61% |

### 3. Group-Based Evaluation

A GroupShuffleSplit strategy was used to prevent identical article titles from appearing in both training and testing sets.

**Linear SVM results:**

| Metric | Score |
|---|---:|
| Accuracy | 99.45% |
| Precision | 99.17% |
| Recall | 99.69% |
| F1-score | 99.43% |

### 4. Temporal Robustness Evaluation

A chronological split was performed using article publication dates.

The temporal evaluation used:

- Earlier articles for training.
- Later articles for testing.
- Duplicate-title overlap removal.

The following experiments were conducted:

| Experiment | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Temporal SVM – Reuters Retained | 99.64% | 99.84% | 99.73% | 99.79% |
| Temporal SVM – Reuters Removed | 98.97% | 99.64% | 99.14% | 99.39% |
| Temporal Unique Titles – Reuters Removed | 99.04% | 99.78% | 99.13% | 99.45% |

**Important observation:** Reuters markers were strongly associated with the Real News class in this dataset. Removing these markers reduced performance, demonstrating the influence of publisher-specific patterns on classification.

The results above are specific to the historical dataset and evaluation procedures. They do not establish generalisation to contemporary news articles.

---

## 🔍 Confusion Matrix

The temporal SVM model without Reuters markers produced the following confusion matrix:

```text
                 Predicted
                 Fake   Real

Actual Fake      1356    27
Actual Real        64  7401
```

This represents:

- True Negatives: 1,356
- False Positives: 27
- False Negatives: 64
- True Positives: 7,401

For this experiment, Real News was treated as the positive class.

---

## 💡 Model Explainability

Linear SVM coefficients were examined to understand which features influenced predictions.

Examples of features associated with Real News:

- `said`
- `washington`
- `presid donald`
- `wednesday`
- `tuesday`

Examples of features associated with Fake News:

- `via`
- `imag`
- `sen`
- `gop`
- `obama`

These features reflect patterns learned from the training dataset. They should not be treated as universal indicators of misinformation.

---

## 🌐 Streamlit Web Application

The final trained model was integrated into a Streamlit application.

### Application features

- News article text input.
- Text preprocessing.
- Fake News and Real News classification.
- Class probability estimates.
- Model confidence score.
- Confidence-based manual-review recommendation.
- Prediction history.
- Clear prediction history functionality.
- CSV export of prediction history.

---

### Live Application

**Streamlit App:** https://fake-and-real-news-detection-lxjponsm73pljjoslxgidx.streamlit.app/.

---

## 📁 Project Structure

```text
Fake-and-Real-News-Detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
│   └── Fake News Detector.ipynb
│
├── models/
│   └── fake_news_detector.joblib
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Harsheet-web/Fake-and-Real-News-Detection.git
```

### 2. Navigate to the project directory

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

---

## 💾 Model Serialization

The final calibrated model was saved using Joblib.

Model location:

```text
models/fake_news_detector.joblib
```

The saved model package contains:

- Calibrated classification pipeline.
- Confidence threshold.
- Label mapping.

The Streamlit application loads this saved model and uses it for predictions.

---

## ⚠️ Limitations

Despite the strong experimental results, several limitations should be considered:

1. The dataset contains historical news articles primarily from 2015–2018.
2. Publisher-specific writing patterns may influence predictions.
3. Dataset labels do not guarantee factual correctness.
4. High model confidence does not guarantee that a prediction is correct.
5. Performance on contemporary articles has not been independently established.
6. The model does not retrieve external sources or perform independent fact-checking.
7. Further evaluation on independent and more recent datasets is necessary.

This application should be treated as an experimental text classification tool, not a replacement for professional fact-checking.

---

## 🚀 Future Improvements

Potential future enhancements include:

- Testing on newer and independent news datasets.
- Improving generalisation across different publishers.
- Experimenting with transformer-based models such as BERT.
- Adding source credibility analysis.
- Improving model explainability.
- Integrating external fact-checking services.
- Improving application analytics and reporting.
- Monitoring performance on new articles.

---

## 🎯 Key Learning Outcomes

Through this project, the following practical skills were developed:

- Exploratory Data Analysis.
- Natural Language Processing.
- Text cleaning and preprocessing.
- TF-IDF feature engineering.
- Supervised Machine Learning.
- Model comparison and evaluation.
- Cross-validation and hyperparameter tuning.
- Data leakage investigation.
- Temporal validation.
- Explainable Machine Learning.
- Probability calibration.
- Confidence-based selective classification.
- Model serialization using Joblib.
- Streamlit application development.
- Git and GitHub version control.
- Cloud application deployment.

---

## 👨‍💻 Author

**Harsheet Raut**

GitHub: [Harsheet-web](https://github.com/Harsheet-web)

Project Repository: [Fake-and-Real-News-Detection](https://github.com/Harsheet-web/Fake-and-Real-News-Detection)

---

⭐ If you find this project useful, consider giving the repository a star.