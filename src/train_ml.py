import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_data


def train_model():
    # Load preprocessed data
    X, y, cat_cols, num_cols = preprocess_data()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing: OneHotEncode categorical, passthrough numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # Model
    model = LogisticRegression(
        max_iter=1000,
        C=0.5,
        penalty ="l2")
    
    tree_model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced"
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    tree_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", tree_model),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", rf_model),
        ]
    )

   

    # Train
    pipeline.fit(X_train, y_train)
    tree_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    

    # Predict
    y_pred = pipeline.predict(X_test)
    tree_preds = tree_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nDecision Tree Classifier:")
    print("Accuracy:", accuracy_score(y_test, tree_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, tree_preds))
    print("\nRandom Forest Classifier:")
    print("Accuracy:", accuracy_score(y_test, rf_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))

     #Feature Importance (Random Forest)

    feature_names = (
        rf_pipeline.named_steps["preprocessing"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
    )
    all_features = np.concatenate([feature_names, num_cols])

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Important Features (Random Forest):")
    for i in indices[:10]:
        print(f"{all_features[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    train_model()
