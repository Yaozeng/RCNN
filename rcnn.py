from keras import backend
from keras.layers import Conv1D, Dense, Input, Lambda, LSTM
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping,TensorBoard
from dataUtil import *
from keras.initializers import Constant
import os

class RCNN:
    def __init__(self):
        self.BASE_DIR = os.getcwd()
        self.MAX_SEQUENCE_LENGTH = 1000
        self.MAX_NUM_WORDS = 20000
        self.EMBEDDING_DIM = 100
        self.VALIDATION_SPLIT = 0.2
        self.NUM_CLASS=20
        self.LSTM_SIZE=200
        self.kenral=100
    def build_model(self,embedding_matrix,num_words):
        document = Input(shape=(None,), dtype="int32")
        left_context = Input(shape=(None,), dtype="int32")
        right_context = Input(shape=(None,), dtype="int32")

        embedding_layer = Embedding(num_words,
                                    self.EMBEDDING_DIM,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        doc_embedding = embedding_layer(document)
        l_embedding = embedding_layer(left_context)
        r_embedding = embedding_layer(right_context)

        # I use LSTM RNNs instead of vanilla RNNs as described in the paper.
        forward = LSTM(self.LSTM_SIZE, return_sequences=True)(l_embedding)  # See equation (1).
        backward = LSTM(self.LSTM_SIZE, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2).
        # Keras returns the output sequences in reverse order.
        backward = Lambda(lambda x: backend.reverse(x, axes=1))(backward)
        together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3).

        semantic = Conv1D(self.kenral, kernel_size=1, activation="tanh")(together)  # See equation (4).

        # Keras provides its own max-pooling layers, but they cannot handle variable length input
        # (as far as I can tell). As a result, I define my own max-pooling layer here.
        pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(self.kenral,))(semantic)  # See equation (5).

        output = Dense(self.NUM_CLASS, input_dim=self.kenral, activation="softmax")(pool_rnn)  # See equations (6) and (7).
        model=Model([document,left_context,right_context],output)
        return model
    def train_model(self):
        GLOVE_DIR = os.path.join(self.BASE_DIR, 'glove.6B')
        TEXT_DATA_DIR = os.path.join(self.BASE_DIR, '20_newsgroup')
        texts,labels_index,labels=load_data_and_labels(TEXT_DATA_DIR)
        x_train, y_train, x_val, y_val, word_index=load_data(texts, labels, self.MAX_NUM_WORDS, self.MAX_SEQUENCE_LENGTH, self.VALIDATION_SPLIT)
        embedding_matrix,num_words=load_pre_trained(GLOVE_DIR, self.MAX_NUM_WORDS, word_index, self.EMBEDDING_DIM)
        model=self.build_model(embedding_matrix,num_words)
        model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])
        x_train_left=np.concatenate((np.full((x_train.shape[0],1),num_words),x_train[:,:-1]),axis=1)
        x_train_right=np.concatenate((x_train[:,1:],np.full((x_train.shape[0],1),num_words)),axis=1)
        x_val_left=np.concatenate((np.full((x_val.shape[0],1),num_words),x_val[:,:-1]),axis=1)
        x_val_right=np.concatenate((x_val[:,1:],np.full((x_val.shape[0],1),num_words)),axis=1)
        model.fit([x_train,x_train_left,x_train_right],y_train,validation_data=([x_val,x_val_left,x_val_right],y_val),batch_size=128,epochs=20,callbacks=[EarlyStopping(monitor='val_loss'),TensorBoard('./logs')])
        model.save("model.h5")
rcnn=RCNN()
rcnn.train_model()


