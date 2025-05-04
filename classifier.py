import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report
import numpy as np
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    df = pd.read_csv("./features_output/articles_with_features.csv")

    to_drop = [
        "Author",
        "Source",
        "NYT",
        "Genre",
        "PubDate",
        "Article Title",
        "Article Text",
        "URL",
    ]
    X = df.drop(to_drop, axis=1).values
    y = df["Source"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("set up test split")

    num_classes = len(le.classes_)

    # model = Sequential(
    #     [
    #         Dense(256, activation="relu", input_shape=(X.shape[1],)),
    #         Dropout(0.3),
    #         Dense(128, activation="relu"),
    #         Dropout(0.3),
    #         Dense(num_classes, activation="softmax"),  # Multiclass output
    #     ]
    # )

    # model.compile(
    #     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )

    # print("begin fitting")

    # model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

    # test_loss, test_accuracy = model.evaluate(X_test, y_test)
    # print("Test accuracy:", test_accuracy)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
