import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Activation
from sklearn.model_selection import train_test_split
import pickle
# Read the dataset from Kaggle and load it into a pandas DataFrame
dataset_path = r"C:\Users\natha\Downloads\archive (2)\ner.csv"
nerDF = pd.read_csv(dataset_path, encoding="ISO-8859-1")

# Preprocess the dataset
nerDF = nerDF[['Sentence #', 'Word', 'Tag']]
nerDF.rename(columns={'Sentence #': 'Sentences'}, inplace=True)
nerDF['Sentences'] = nerDF['Sentences'].ffill()
sentenceList = [(list(zip(group['Word'], group['Tag']))) for _, group in nerDF.groupby('Sentences')]

# Tokenize words and tags
all_words = [word for sentence in sentenceList for word, tag in sentence]
all_tags = [tag for sentence in sentenceList for word, tag in sentence]

# Create word-to-index and tag-to-index mappings
word_to_index = {"<PAD>": 0, "<OOV>": 1}
for word in all_words:
    if word not in word_to_index:
        word_to_index[word] = len(word_to_index)

tag_to_index = {tag: index for index, tag in enumerate(set(all_tags))}

# Convert words and tags to sequences of integers
X = [[word_to_index[word] for word, _ in sentence] for sentence in sentenceList]
y = [[tag_to_index[tag] for _, tag in sentence] for sentence in sentenceList]



# Pad sequences
max_len = max(len(sentence) for sentence in X)
X = pad_sequences(X, padding='post', maxlen=max_len)
y = pad_sequences(y, padding='post', maxlen=max_len)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile the bi-directional LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(word_to_index), output_dim=50, input_length=max_len, mask_zero=False))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dense(len(tag_to_index), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=5)

filename = 'finalizedNER_model.sav'
pickle.dump(model, open(filename, 'wb'))