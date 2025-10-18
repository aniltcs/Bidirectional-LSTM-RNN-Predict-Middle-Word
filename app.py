import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model and objects
model = load_model("food_predictor_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get max_len from training tokenizer
## max_len = max(len(x) for x in tokenizer.texts_to_sequences(tokenizer.word_index))
max_len = model.input_shape[1]

# Prediction function
def predict_missing_food(sentence):
    sentence = sentence.replace('_', 'something')
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    index = np.argmax(pred)
    return label_encoder.inverse_transform([index])[0]

# Streamlit UI
st.title("Missing Food Predictor")
st.write("Fill in the blank with the correct food based on context.")

user_input = st.text_input("Enter a sentence (use _ for the missing food):", 
                           "Rahul eats _ in Kolkata")

if st.button("Predict"):
    prediction = predict_missing_food(user_input)
    st.success(f" Predicted Food: **{prediction}**")
