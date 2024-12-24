import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model= load_model('RNN.h5')

word_index=imdb.get_word_index()


def preprocess_text(text):
    sent=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in sent]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

st.title("IMDB DATA SENTIMENT")
st.write("Enter Feedback of Movie:")
input= st.text_input("Movie Review Here", value="")

if st.button('Classify'):
    
    padded=preprocess_text(input)
    predicted=model.predict(padded)
    sentiment='Positive' if predicted[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {predicted[0][0]}')