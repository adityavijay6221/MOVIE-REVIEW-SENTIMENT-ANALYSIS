## 1. Problem Statement
The goal of this project is to classify movie reviews from the IMDB dataset as Positive or Negative using a Recurrent Neural Network (RNN).
Sentiment analysis is a key NLP task widely used in applications like customer feedback analysis, product review filtering, and social media monitoring.
Our objective is to build a model capable of understanding context and sequence in text data to make accurate predictions.

## 2. Exploratory Data Analysis (EDA) & Preprocessing
Dataset: IMDB reviews dataset (from tensorflow.keras.datasets.imdb)
Each review is pre-encoded as a sequence of integers representing words.
## Key Steps:
- Data Loading: Loaded IMDB dataset with 25,000 training and 25,000 testing samples.
- Tokenization: IMDB dataset provides word-to-index mapping for 10,000 most frequent words.
- Sequence Padding: All reviews were padded to a uniform length (maxlen = 500) using pad_sequences.
- Decoding Reviews: A helper function was created to decode integer-encoded reviews back to readable text for interpretability.
- Class Distribution: Dataset is balanced with equal positive and negative reviews.
- Text Cleaning: Lowercasing, removing unknown tokens, and handling out-of-vocabulary words.
  
## 3. Model Architecture & Training
The model was built using TensorFlow/Keras Sequential API.
It employs an Embedding layer followed by a SimpleRNN and Dense output layer.
| Layer               | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| **Embedding**       | Converts integer-encoded words into dense vectors of fixed size                 |
| **SimpleRNN**       | Captures sequential patterns and dependencies across word sequences             |
| **Dense (Sigmoid)** | Outputs a probability value between 0 and 1 for binary sentiment classification |

## Training Details:

- Loss Function: binary_crossentropy
- Optimizer: adam
- Metrics: accuracy
- Epochs: 10–15 (based on convergence)
- Batch Size: 64
- Validation Split: 20% of training data used for validation

The model (model_rnn.h5) was saved post-training for deployment.

## 4. Evaluation Process & Insights
Evaluation Metrics:
- Accuracy: ~86–88% on test data
- Loss Curves: Training and validation losses stabilized, indicating good generalization.
- Precision & Recall: Balanced, showing minimal bias toward any class.

 ## Key Insights:
- RNN effectively captured sequential dependencies, improving sentiment understanding.
- Longer review texts occasionally reduced performance due to vanishing gradients — LSTM/GRU could further improve results.
- Model performs well on standard reviews but may misclassify sarcastic or ambiguous reviews.

## 5. Deployment & Impact
The trained model was deployed as an interactive Streamlit web application (app.py).

## App Features:
- User Input: Accepts raw movie reviews from users.
- Real-Time Prediction: Displays sentiment (Positive or Negative) and prediction confidence score.
- Backend: Loads model_rnn.h5 for inference and preprocesses text dynamically.
- Frontend: Clean Streamlit interface for usability.
## Impact:
This project demonstrates a complete NLP workflow — from model training to deployment — and showcases the power of deep learning for real-world sentiment analysis applications.
It can be extended to analyze social media comments, product reviews, or customer feedback.
  

