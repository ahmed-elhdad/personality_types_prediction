import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("../data/cleaned_data.csv")
numeric_features = ["words_count", "text_length"]
text_features = "cleaned_posts"
X = data.drop("type", axis=1)
y = data["type"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


transformers = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "text",
            TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                min_df=5,
                max_df=0.8,
                max_features=10000,
            ),
            text_features,
        ),
    ],
    remainder="drop",
)
pipeline = Pipeline(
    [
        ("prep", transformers),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42)),
    ]
)
pipeline.fit(X_train, y_train)
# Getting start Grid Search
param_grid = {
    "prep__text__max_features": [5000, 10000],
    "prep__text__ngram_range": [(1, 2)],
    "clf__C": [
        0.1,
        1,
    ],
}
grid = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, scoring="f1_weighted")
grid.fit(X_train, y_train)
# Predict
y_pred = grid.predict(X_test)
# Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(grid.score(X_test, y_test) * 100)
import joblib

joblib.dump(grid, "logistic_regression_pipline.pkl")
