import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import streamlit as st

with open('Tokenizer_File.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_len):
    tokens = tokenizer.texts_to_sequences([text])[0]

    # Keep only the last (max_len_sequences - 1) tokens
    if len(tokens) >= max_len:
        tokens = tokens[-(max_len - 1) : ]

    tokens = pad_sequences(
        [tokens],
        maxlen=max_len - 1,
        padding='pre'
    )

    predicted = model.predict(tokens, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

model = load_model('next_word_predicter_lstm.h5')

st.title("Next Word Prediction With LSTM")

input_data = st.text_input("Enter the sequence of Words")

if st.button("Predict The Next Word"):
    max_len_sequence = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_len_sequence)
    st.write(f'Next word: {next_word}')