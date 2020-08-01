import os
import configparser
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, fields

from preprocessor import Preprocessor
from model import lstm_classifier_model


# parameter
config = configparser.ConfigParser()
config.read('config.ini')
data_dir = config['config']['data_dir']
max_len = int(config['config']['max_len'])
hidden_size = int(config['config']['hidden_size'])

# create preprocessor
preprocessor = Preprocessor(data_dir, max_len)
preprocessor.load_vocab()
preprocessor.load_labels()

vocab_size = preprocessor.get_vocab_size()
num_labels = preprocessor.get_num_labels()
labels = preprocessor.get_labels()

# create model
model = lstm_classifier_model(vocab_size, max_len, hidden_size, num_labels)

# load checkpoint
model.load_weights(os.path.join(data_dir, 'checkpoint'))

# flask setting
app = Flask(__name__)
api = Api(app, version='1.0', title='simple lstm classification')
ns  = api.namespace('predict')

text_api_model = api.model(name='text', model={
    'text': fields.String(required=True, description='text', example='hello')
})

@ns.route('/')
class Predict(Resource):
    @api.expect(text_api_model)
    def post(self):
        text = request.json['text']
        feature = preprocessor.make_feature(text)
        prediction = model.predict([feature])
        result = labels[tf.math.argmax(prediction[0])]
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config['service']['port'])