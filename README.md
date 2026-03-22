# Personality Types Prediction

A machine learning project that predicts Myers-Briggs Type Indicator (MBTI) personality types based on text input using a Logistic Regression model.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Analysis and Model](#data-analysis-and-model)
- [Accuracy](#accuracy)
- [Libraries Used](#libraries-used)
- [App Screenshot](#app-screenshot)
- [Contributing](#contributing)
- [License](#license)

## Description
This project aims to classify personality types into one of the 16 MBTI types based on textual data. The model is trained on cleaned text posts and uses features like word count and text length along with TF-IDF vectorization of the text content.

The MBTI types are combinations of four dichotomies:
- Introversion (I) vs Extroversion (E)
- Intuition (N) vs Sensing (S)
- Thinking (T) vs Feeling (F)
- Judging (J) vs Perceiving (P)

## Features
- Data cleaning and preprocessing pipeline
- Feature engineering (word count, text length)
- TF-IDF vectorization for text features
- Logistic Regression with hyperparameter tuning via GridSearchCV
- Streamlit web app for easy prediction
- Model serialization with joblib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personality_types_prediction.git
   cd personality_types_prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preparation:**
   Run the data cleaning script:
   ```bash
   python src/clean_data.py
   ```
   This processes `data/personality_types.csv` and saves the cleaned data to `data/cleaned_data.csv`.

2. **Model Training:**
   Run the model training script:
   ```bash
   python src/Logistic_Regression_model.py
   ```
   This trains the model and saves it as `src/logistic_regression_pipline.pkl`.

3. **Run the App:**
   Start the Streamlit app:
   ```bash
   streamlit run src/main.py
   ```
   Open your browser to the provided URL and enter text to predict the personality type.

## Project Structure
```
personality_types_prediction/
├── data/
│   ├── personality_types.csv    # Raw dataset with personality types and posts
│   └── cleaned_data.csv         # Processed dataset after cleaning
├── src/
│   ├── clean_data.py            # Data cleaning and preprocessing script
│   ├── Logistic_Regression_model.py  # Model training script
│   ├── main.py                  # Streamlit web app
│   └── logistic_regression_pipline.pkl  # Trained model (generated)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### File Descriptions
- **`src/clean_data.py`**: 
  - Loads raw data from `data/personality_types.csv`
  - Cleans text by removing URLs, punctuation, digits, and converting to lowercase
  - Splits text on delimiters and joins back
  - Creates features: `words_count` (number of words), `text_length` (character count)
  - Saves cleaned data to `data/cleaned_data.csv`

- **`src/Logistic_Regression_model.py`**:
  - Loads cleaned data
  - Splits into train/test sets (70/30 split, stratified)
  - Creates a pipeline with ColumnTransformer for numeric and text features
  - Numeric features: StandardScaler on `words_count` and `text_length`
  - Text features: TF-IDF vectorizer (ngrams 1-2, stop words, min_df=5, max_df=0.8, max_features=10000)
  - Logistic Regression with class balancing
  - Performs GridSearchCV on hyperparameters (max_features, ngram_range, C)
  - Evaluates with classification report, confusion matrix, and accuracy
  - Saves the best model to `src/logistic_regression_pipline.pkl`

- **`src/main.py`**:
  - Loads the trained model
  - Creates a Streamlit interface with title and text area
  - On button click, creates a DataFrame with input text, word count, and text length
  - Predicts the personality type and displays it
  - Shows the key for MBTI letters

## Data Analysis and Model
- **Dataset**: Contains text posts labeled with MBTI personality types
- **Preprocessing**: Text cleaning removes noise, feature engineering adds numeric features
- **Model**: Logistic Regression in a scikit-learn pipeline
- **Hyperparameter Tuning**: Grid search on TF-IDF max_features and regularization parameter C
- **Evaluation**: F1-weighted score, classification report, confusion matrix

## Accuracy
The model achieves an accuracy of 64% on the test set.

## Libraries Used
This project uses the following Python libraries:

<p align="center">
  <img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="Pandas" width="100" height="100">
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Scikit-learn" width="100" height="100">
  <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" alt="Streamlit" width="100" height="100">
  <img src="https://joblib.readthedocs.io/en/latest/_static/joblib_logo.svg" alt="Joblib" width="100" height="100">
</p>

- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms, preprocessing, and evaluation
- **Streamlit**: Web app framework for the prediction interface
- **Joblib**: Model serialization and loading

## App Screenshot
![App Screenshot](images/app_screenshot.png)

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.</content>
<parameter name="filePath">d:\work\GitHub\personality_types_prediction\README.md