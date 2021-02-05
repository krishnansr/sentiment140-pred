import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def process_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


X = []
movie_reviews = pd.read_csv("test_data.csv")
movie_reviews.isnull().values.any()
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(process_text(sen))

y = movie_reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = 290419  # change this to "len(tokenizer.word_index) + 1"
max_len = 100        # optional and can be set to desired val
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

w2v_model = KeyedVectors.load("model.w2v", mmap='r')
W2V_SIZE = 300

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], trainable=False)


model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.load_weights('model.h5')
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

score = model.evaluate(X_test, y_test, batch_size=1)

print("ACCURACY:", score[1])
print("LOSS:", score[0])


