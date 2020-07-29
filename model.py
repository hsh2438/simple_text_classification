import tensorflow as tf


def lstm_classifier_model(vocab_size, max_len, hidden_size, num_labels):
    
    # model define
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size)),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(num_labels, activation='softmax')
    ])

    # model compile
    model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model