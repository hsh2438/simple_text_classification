import os

class Data:

    def __init__(self, data_dir, max_len):
        self.data_dir = data_dir
        self.max_len = max_len

        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.vocab = [self.PAD, self.UNK]
        self.labels = []

    def __load_file__(self, data_file):
        data = []
        with open(os.path.join(self.data_dir, data_file), 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                data.append(line.strip().split('\t'))
        return data

    # load train and test data
    def load_data(self):
        self.train_data = self.__load_file__('train.tsv')
        self.test_data = self.__load_file__('test.tsv')

    # load vocab from vocab.txt
    def load_vocab(self):
        vocab.extend(__load_file__('vocab.txt'))
    
    # load labels from labels.txt
    def load_labels(self):
        labels.extend(__load_file__('labels.txt'))

    # make vocab and labels from train data
    def make_vocab(self):
        for line in self.train_data:
            text, label = line[0], line[1]
            for token in self.tokenize(text):
                if not token in self.vocab:
                    self.vocab.append(token)
            if not label in self.labels:
                self.labels.append(label)

    # tokenize
    def tokenize(self, text):
        return text.split()
    
    # vectorize text and label
    def make_features(self):

        def __make_features__(data):
            features = []
            labels = []
            for line in self.train_data:
                text, label = line[0], line[1]
                feature = []
                for token in self.tokenize(text):
                    if not token in self.vocab:
                        feature.append(self.vocab.index(self.UNK))
                    else:
                        feature.append(self.vocab.index(token))
                while len(feature) < self.max_len:
                    feature.append(self.vocab.index(self.PAD))
                features.append(feature)
                assert label in self.labels
                labels.append(self.labels.index(label))

            return features, labels

        self.train_features, self.train_labels = __make_features__(self.train_data)
        self.test_features, self.test_labels = __make_features__(self.test_data)
