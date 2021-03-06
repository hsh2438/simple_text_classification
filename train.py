import os
import configparser
import tensorflow as tf

from preprocessor import Preprocessor
from model import lstm_classifier_model


# parameter
config = configparser.ConfigParser()
config.read('config.ini')
data_dir = config['config']['data_dir']
max_len = int(config['config']['max_len'])
hidden_size = int(config['config']['hidden_size'])
epochs = int(config['train']['epochs'])

# create preprocessor and data loading
preprocessor = Preprocessor(data_dir, max_len)
preprocessor.load_data()
preprocessor.make_vocab()
preprocessor.make_train_features()
preprocessor.make_test_features()

train_features, train_labels = preprocessor.get_train_features()
test_features, test_labels = preprocessor.get_test_features()
vocab_size = preprocessor.get_vocab_size()
num_labels = preprocessor.get_num_labels()

# create model
model = lstm_classifier_model(vocab_size, max_len, hidden_size, num_labels)

# training
model.fit(train_features, train_labels, epochs=epochs, shuffle=True)

# evaluation
test_loss, test_acc = model.evaluate(test_features, test_labels)

# save checkpoint
model.save_weights(os.path.join(data_dir, 'checkpoint'))


print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
