from main import *
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Activation
from sklearn.model_selection import train_test_split
import pickle

loaded_model = pickle.load(open('finalizedNER_model.sav', 'rb'))

example_paragraph = "Nathan enjoys Natural Language Processing while living in California and thinking about Google. Nathan will be attending the University of Urbana Champaign soon, which is 2 hours south of Chicago, Illinois. Nathan, also a fan of Nvidia, has thought of a random name Bob who works with Twitch in Seattle."

# Preprocess the example paragraph
example_sentences = example_paragraph.split(". ")
example_words_list = [sentence.split() for sentence in example_sentences]
example_sequence_list = [[word_to_index.get(word, word_to_index["<OOV>"]) for word in words] for words in example_words_list]
example_padded_sequence_list = pad_sequences(example_sequence_list, padding='post', maxlen=max_len)

# Pred
predicted_indices_list = loaded_model.predict(np.array(example_padded_sequence_list))
predicted_tags_list = [[list(tag_to_index.keys())[list(tag_to_index.values()).index(idx)] for idx in sentence_indices if idx != 0]
                       for sentence_indices in np.argmax(predicted_indices_list, axis=-1)]

# Print results
print("Example Paragraph:")
for i, sentence in enumerate(example_sentences):
    print("Sentence:", sentence)
    print("Predicted Tags:", predicted_tags_list[i])
    print()