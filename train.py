import os
import numpy as np
import tensorflow as tf

from data_util import Data


max_len = 64
hidden_size = 64

data = Data('data', max_len)
data.load_data()
data.make_vocab()
data.make_features()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(data.vocab), max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size)),
    tf.keras.layers.Dense(hidden_size, activation='relu'),
    tf.keras.layers.Dense(len(data.labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data.train_features, data.train_labels, epochs=30, shuffle=True)

test_loss, test_acc = model.evaluate(data.test_features, data.test_labels)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))