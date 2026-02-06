import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import layers

from preprocess import preprocess_data

def train_dl_model():
    #Load data 
    X,y, cat_cols, num_cols = preprocess_data()

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to dense (TensorFlow needs dense input)
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

    # Build neural network
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train_processed.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    model.fit(
        X_train_processed,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test_processed, y_test)
    print("\nTest Accuracy:", accuracy)


if __name__ == "__main__":
    train_dl_model()