# app_cnn.py
# A *very simple* Streamlit app to demonstrate how a CNN works using MNIST (built-in Keras dataset).
# It will:
# 1) Load MNIST
# 2) Build a tiny CNN
# 3) Train it and show logs + history (loss/accuracy curves)
# 4) Evaluate on test set
# 5) Let you pick a random test image and see the model's prediction

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple CNN Demo (MNIST)", layout="wide")
st.title("🧠 Simple CNN Demo (MNIST)")

st.write(
    "This app trains a small Convolutional Neural Network (CNN) on the **MNIST** handwritten digits dataset "
    "and lets you test predictions on random images."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Training Settings")
epochs = st.sidebar.slider("Epochs", 1, 10, 3)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], index=2)

use_small_subset = st.sidebar.checkbox("Use small subset for faster demo", value=True)
subset_train = st.sidebar.slider("Train subset size", 2000, 20000, 6000, 1000, disabled=not use_small_subset)
subset_test = st.sidebar.slider("Test subset size", 500, 5000, 1500, 500, disabled=not use_small_subset)

st.sidebar.markdown("---")
seed = st.sidebar.number_input("Random seed", 0, 999999, 42, 1)

# -----------------------------
# Helper: cache data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

# -----------------------------
# Helper: build model
# -----------------------------
def build_cnn(lr: float):
    # A small CNN:
    # - Conv detects local patterns (edges/curves)
    # - MaxPool shrinks the image while keeping important signals
    # - Dense layers decide which digit it is
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    
