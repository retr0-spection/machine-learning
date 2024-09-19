import matplotlib.pyplot as plt
import numpy as np

k = 1
class NaiveBayesClassifier():
    def __init__(self) -> None:
        self.class_names = []
        self.training_count = 0
        self.feature_labels = []
        self.word_list = []
        self.likelihood_table = None
        self.freq_table = None

    def _get_unique_word_list(self, X):
        word_list = set()
        for index, row in X.iterrows():
            for i in self.feature_labels:
                word_list.add(row.iloc[i])


        return word_list

    def _get_class_count(self, data):
        count = dict()
        for index, row in data.iterrows():
            try:
                count[row.iloc[0]] += 1
                self.training_count += 1
            except Exception:
                count[row.iloc[0]] = 1
                self.training_count += 1


        self.class_counts = count
        return count

    def train_model(self, X_train,y_train):
        self.feature_labels = [i for i in range(X_train.shape[1])]
        unique_word_list = [i for i in self._get_unique_word_list(X_train)]
        self.word_list = unique_word_list
        vocab_size = len(self.feature_labels)
        self.class_count = self._get_class_count(y_train)


        #generate frequency table
        freq_table = np.zeros((len(self.class_count.keys()), X_train.shape[1], len(unique_word_list)), dtype=int)

        for index, row in X_train.iterrows():
            row = row.to_list()
            for pixel_index, pixel_value in enumerate(row):
                label = y_train.loc[index].iloc[0]
                freq_table[label, pixel_index, pixel_value] += 1


        self.freq_table = freq_table

        #generate likelihood prob table
        likelihood_tables = np.zeros((len(self.class_count.keys()), X_train.shape[1], len(unique_word_list)), dtype=float)
        

        for class_name, class_count in self.class_count.items():
            total_count_class = np.sum(freq_table[class_name]) + vocab_size

            for pixel_index in self.feature_labels:
                for word in self.word_list:
                    likelihood_tables[class_name, pixel_index, word] = (freq_table[class_name, pixel_index, word] + k) / total_count_class


        self.likelihood_table = likelihood_tables


        return likelihood_tables


    def _get_label(self, row):
        prob_given_class = np.zeros(len(self.class_count.keys()))
        pixels = row.to_list()

        for pixel_index, pixel_value in enumerate(pixels):
                for i, class_name in enumerate(self.class_count.keys()):
                    prob_given_class[i] += np.log(self.likelihood_table[class_name, pixel_index, pixel_value])


        # multiply class priors
        #
                    
        total = 0
        for i, class_name in enumerate(self.class_count.keys()):
            total += np.sum(self.freq_table[class_name])

        for i, class_name in enumerate(self.class_count.keys()):
            prob_given_class[i] += np.log(np.sum(self.freq_table[class_name])/total)


        return  np.where(prob_given_class == max(prob_given_class))[0][0]


    def classify(self, X, y):
        results = []

        for index, row in X.iterrows():
            result = self._get_label(row)
            results.append(result)

        return results
