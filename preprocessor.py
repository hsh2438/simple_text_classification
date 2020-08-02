import os

class Preprocessor:

    def __init__(self, data_dir, max_len):
        self.data_dir = data_dir
        self.max_len = max_len

        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.vocab = [self.PAD, self.UNK]
        self.labels = []

    def __load_file__(self, data_file, splited=True):
        data = []
        with open(os.path.join(self.data_dir, data_file), 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                if splited:
                    data.append(line.strip().split('\t'))
                else:
                    data.append(line.strip())
        return data

    def __save_file__(self, data_file, data):
        with open(os.path.join(self.data_dir, data_file), 'w', encoding='utf-8') as fw:
            for line in data:
                fw.write(line+'\n')

    # load train and test data
    def load_data(self):
        self.train_data = self.__load_file__('train.tsv')
        self.test_data = self.__load_file__('test.tsv')

    # load vocab from vocab.txt
    def load_vocab(self):
        self.vocab = self.__load_file__('vocab.txt', splited=False)
    
    # load labels from labels.txt
    def load_labels(self):
        self.labels = self.__load_file__('labels.txt', splited=False)

    # make vocab and labels from train data
    def make_vocab(self):
        for line in self.train_data:
            text, label = line[0], line[1]
            for token in self.tokenize(text):
                if not token in self.vocab:
                    self.vocab.append(token)
            if not label in self.labels:
                self.labels.append(label)

        self.__save_file__('vocab.txt', self.vocab)
        self.__save_file__('labels.txt', self.labels)

    # tokenize
    def tokenize(self, text):
        return text.split()
    
    # vectorize text
    def make_feature(self, text):
        feature = []
        for token in self.tokenize(text):
            if not token in self.vocab:
                feature.append(self.vocab.index(self.UNK))
            else:
                feature.append(self.vocab.index(token))
        while len(feature) < self.max_len:
            feature.append(self.vocab.index(self.PAD))
        return feature

    def __make_features__(self, data):
        features = []
        labels = []

        for line in self.train_data:
            text, label = line[0], line[1]
            feature = self.make_feature(text)
            features.append(feature)
            assert label in self.labels
            labels.append(self.labels.index(label))

        return features, labels

    # vectorize train set
    def make_train_features(self):
        self.__train_features, self.__train_labels = self.__make_features__(self.train_data)

    # vectorize test set
    def make_test_features(self):
        self.__test_features, self.__test_labels = self.__make_features__(self.test_data)
    
    # return vocab size
    def get_vocab_size(self):
        return len(self.vocab)
    
    # return number of labels
    def get_num_labels(self):
        return len(self.labels)
    
    def get_labels(self):
        return self.labels

    # return train features
    def get_train_features(self):
        return self.__train_features, self.__train_labels

    # return test features
    def get_test_features(self):
        return self.__test_features, self.__test_labels
