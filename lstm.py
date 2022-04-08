import numpy as np
import tensorflow as tf
from keras.datasets import imdb

#top 1000 words. Can change this to get a bigger vocabulary (more words)
n_unique_words = 1000
#timesteps for LSTM
timestep = 200
#how many features will the embedding contain (used for batch size in model.fit)
numFeatures = 128
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=n_unique_words)

#pad test and training data with 0 (make sure same length)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=timestep)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=timestep)

#format data so we can use it
y_train = np.array(y_train)
y_test = np.array(y_test) 

#create sequential model
model = tf.keras.models.Sequential()
#create Embedding layer - takes input and makes it easy to use with LSTM
#can "manually" do this yourself if you don't want to use embedding
model.add(tf.keras.layers.Embedding(n_unique_words, numFeatures, input_length=timestep))
#uncomment link below for vanilla RNN
#model.add(tf.keras.layers.SimpleRNN(64))
#uncomment line below for bidirectional LSTM
#model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
#uncomment line below to add another LSTM layer - return_sequences=True passes all hidden states to the next layer which is needed. Setting this to true is needed for all 
#model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.summary()

model.fit(x_train, y_train,
           batch_size=numFeatures,
           epochs=5,
           validation_data=[x_test, y_test])