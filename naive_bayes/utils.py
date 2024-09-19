import random
import numpy as np


class NaiveBayesClassifier():
    def __init__(self):
        self.joint_probs = None
        self.word_list = []
        self.classes = []
        self.class_priors = {}
        self.class_counts = 0

    def parse_input(self, filename):
        data = []

        with open(filename, 'r') as file:
            data = file.readlines()


        #remove crlf
        data = [i.rstrip('\n') for i in data]

        return data

    def split_train_test(self, data, test_no):
        #randomise data
        split_size = test_no
        randomized_data = []
        while split_size:
            _size = len(data)
            assert split_size <= _size
            rand_index = random.randint(0, _size-1)
            randomized_data.append(data.pop(rand_index))

            #decrement
            split_size -= 1

        return np.array(randomized_data), np.array(data)

    def split_feature_label(self, data):
        y_labels = []
        x_labels = []
        for item in data:
            _item = item.split(' ')
            y_label = _item[0]
            y_labels.append(y_label)

            x_labels.append(item.split(y_label + ' ')[1])

        return np.array(y_labels), np.array(x_labels)

    def probabilities_of_classes(self, data):
        probabilities = dict()
        total = len(data)
        for item in data:
            try:
                probabilities[item] += 1/total
            except Exception:
                probabilities[item] = 1/total
        self.class_priors = probabilities
        return probabilities


    def _get_unique_word_list(self, X):
        word_list = set()
        for review in X:
            for word in review.split(" "):
                if word != '' and len(word) > 2:
                    word_list.add(word)

        self.word_list = word_list

        return word_list

    def _get_class_count(self, data):
        count = dict()
        for item in data:
            try:
                count[item] += 1
            except Exception:
                count[item] = 1

        self.class_counts = count
        return count

    def train(self, X, y):
        self.probabilities_of_classes(y)
        self.joint_probabilities(X, y)

    def _encode_input(self, X):
        sentence = X.split(" ")
        sentence = [_ for _ in sentence if _ != '']
        word_list = [i for i in self.word_list]
        encoded_sentence = []
        unknown_symbols = []
        for word in sentence:
            try:
                if len(word) > 2: #discard stop words
                    encoded_sentence.append(word_list.index(word))
            except Exception:
                unknown_symbols.append(-1)

        encoded_sentence = [1 if i in encoded_sentence else 0 for i in range(len(word_list))]


        return encoded_sentence, unknown_symbols

    def _generate_confusion_matrix(self, pred, actual_labels):
        classes = self.classes

        TP = TN = FP = FN = 0

        for p, a in zip(pred, actual_labels):
            if p == '1' and a == '1':
                TP += 1
            elif p == '-1' and a == '-1':
                TN += 1
            elif p == '1' and a == '-1':
                FP += 1
            elif p == '-1' and a == '1':
                FN += 1


        return TP, TN, FP, FN

    def classify(self, X, actual=None, error=False):
        #encode X
        print(X)
        results = []
        for _input in X:
            encoded_sentence, unknown_symbols = self._encode_input(_input)
            k = 1

            prob_given_class = np.ones(len(self.classes))

            #multiply probabilties from table as if they were independent for each class (bayes theorem)
            for i, _class in enumerate(self.classes):
                for j, encoding in enumerate(encoded_sentence):
                    if encoding:
                        prob_given_class[i] *= self.joint_probs[j, i]
                    else:
                        prob_given_class[i] *= (1 - self.joint_probs[j, i])



            #multiply unknown symbols (laplace smooting)

            # for i, _class in enumerate(self.classes):
            #     for j, encoding in enumerate(unknown_symbols):
            #         prob_given_class[i] *=  (1 + k) /(self.class_counts[_class] + len(self.word_list))



            #multiply priors
            for i, _class in enumerate(self.classes):
                prob_given_class[i] *= self.class_priors[_class]



            #normalize divide by P(x)
            prob_x = 0
            for i, _class in enumerate(self.classes):
                prob_x += prob_given_class[i] * self.class_priors[_class]


            for i, _class in enumerate(self.classes):
                prob_given_class[i] /= prob_x

            print(prob_given_class)



            results.append(self.classes[np.where(prob_given_class == max(prob_given_class))[0][0]])

        if error:
            TP, TN, FP, FN = self._generate_confusion_matrix(results, actual)
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            print(accuracy)


        return results








    def joint_probabilities(self, X, y):
        # lock onto a symbol (word) and find how many times/probability it occurs in classes

        word_list = self._get_unique_word_list(X)

        vocab_size = len(word_list)


        #get class count
        class_counts = self._get_class_count(y)
        class_values = [i for i in class_counts.keys()]
        self.classes = class_values

        freq_table = np.zeros((len(word_list), len(class_counts.keys())))





        #for laplacian smoothing
        k = 1
        for j, _review in enumerate(X):
            _review = _review.split(" ")
            _review = [_ for _ in _review if _ != '']
            for i, word in enumerate(word_list):
                if word in _review:
                        #check which class
                        class_index =  class_values.index(y[j])
                        if freq_table[i, class_index] == 0:
                            freq_table[i, class_index] += (1 + k) /(class_counts[y[j]] + vocab_size)
                        else:
                            freq_table[i, class_index]  += (1) /(class_counts[y[j]] + vocab_size)


        #laplacian smoothing over missing words (no zero probability, see Zero-Frequency problem)
            for i, word in enumerate(word_list):
                for j, _class in enumerate(class_values):
                    if freq_table[i, j] == 0:
                        freq_table[i, j] = (k) /(class_counts[y[j]] + vocab_size)




        self.joint_probs = freq_table

        return freq_table
