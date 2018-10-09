from rcnn import RCNN
import os
from dataUtil import load_data_and_labels,load_data,load_pre_trained
from nltk import sent_tokenize,word_tokenize
import numpy as np
BASE_DIR = os.getcwd()
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
NUM_CLASS=20
LSTM_SIZE=200
kenral=100


GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
texts, labels_index, labels = load_data_and_labels(TEXT_DATA_DIR)
x_train, y_train, x_val, y_val, word_index = load_data(texts, labels, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH,
                                                       VALIDATION_SPLIT)
embedding_matrix, num_words = load_pre_trained(GLOVE_DIR, MAX_NUM_WORDS, word_index, EMBEDDING_DIM)

model=RCNN.build_model(embedding_matrix,word_index)
text="i love nlp"
sentences = [word_tokenize(sent) for sent in sent_tokenize(text)]
sentences2id=[word_index[token] for token in sentences]
pre=model.predict(sentences2id)
pre=np.argmax(pre,axis=0)
for name in labels_index.iterKeys():
    if labels_index[name]==pre:
        print(name)


