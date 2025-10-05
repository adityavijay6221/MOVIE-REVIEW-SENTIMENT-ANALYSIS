import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
model=load_model('model_rnn.h5')

word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}


#DEFINING HELPER FUNCTIONS
#1. DECODING REVIEWS
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

#2. PRE_PROCESS TEXT
def pre_process(text):
  words=text.lower().split() # Splitting sentence into words and lowering the case
  encoded_review=[word_index.get(word,2) + 3 for word in words]
  padded_review=pad_sequences([encoded_review],maxlen=500)
  return padded_review



#FUNCTION TO PREDICT SENTIMENT
def predict_sentiment(review):
  pre_processed_input=pre_process(review)
  prediction=model.predict(pre_processed_input)
  sentiment='Positive' if prediction[0][0]>0.8 else 'Negative'
  return sentiment, prediction[0][0]


#STREAMLIT APP
st.title('IMDB movie analysis')
st.write('Enter the review to classify as Positive or Negative')

#TEXT BOX
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    processed_input=pre_process(user_input)
    prediction=model.predict(processed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
    
else:
    st.write('Please enter a review')
        
