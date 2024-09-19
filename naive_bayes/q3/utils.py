import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayesGaussian():
    def __init__(self):
        self.features = []
        self.mean_table = None
        self.variance_table = None
        self.class_names = []
        self.priors = []
        self.likelihood_table = None

    def _calculate_mean(self, data):
        return np.sum(data)/data.shape[0]

    def _calculate_variance(self, data, mean):
        xi_s = data.to_list()
        _sum = np.sum(np.array([(i - mean)**2 for i in xi_s]))

        _sum /= (len(xi_s) - 1)

        return _sum


    def _draw_histogram(self, X, y):
        class_names = y['class'].unique()
        for class_name in class_names:
            features = X.columns.to_list()
            for feature in features:
                plt.hist(X[feature], bins=30, color='skyblue', edgecolor='black')
                # Adding labels and title
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.title(f'Histogram showing {feature} for class:{class_name}')
                plt.savefig(f'hist-{feature}-{class_name}.png')


    def _generate_likelihood_feature_given_class(self, i,feature, class_name, data):
        constant = (1/np.sqrt(2*np.pi * self.variance_table[i, class_name]))
        expon_inner_1 = -0.5*(data[feature] - self.mean_table[i, class_name])**2
        expon_inner_2 = self.variance_table[i, class_name]
        expon = np.exp(expon_inner_1/expon_inner_2)
        _value =  constant * expon
        return np.log(_value)




    def fit(self, X_train, y_train):
        self.features = X_train.columns.to_list()
        self.class_names = y_train['class'].unique()
        self.priors = [y_train[y_train['class'] == class_value].shape[0]/y_train.shape[0] for class_value in self.class_names]

        data = X_train.join(y_train)
        #likelihood vector/table
        #generate mean and variance given a class
        self.mean_table = np.zeros((len(self.features), len(self.class_names)))
        self.variance_table = np.zeros((len(self.features), len(self.class_names)))


        for class_name in self.class_names:
            #generate mean table for each feature given a class
            _table = data[data['class'] == class_name]

            for i, feature in enumerate(self.features):
                _mean = self._calculate_mean(_table[feature])
                self.mean_table[i, class_name] = _mean



            #generate variance table for each feature

            for i, feature in enumerate(self.features):
                _var = self._calculate_variance(_table[feature], self.mean_table[i, class_name])
                self.variance_table[i, class_name] = _var




    def classify(self, X_test):
        results = []



        #generate likelihood table
        self.likelihood_table = np.zeros((len(self.features), len(self.class_names)))

        for index, row in X_test.iterrows():
            probs = np.zeros((len(self.class_names)))
            for class_name in self.class_names:
                for i, feature in enumerate(self.features):
                    self.likelihood_table[i, class_name] = self._generate_likelihood_feature_given_class(i, feature, class_name, row)


            # calculate prob of data given class using likelihood table and class prior, assuming independence of data naive bayes condition

            prob_given_class = np.zeros((len(self.class_names)))

            for class_name in self.class_names:
                prob_given_class[class_name] += np.sum(self.likelihood_table[:,class_name]) + np.log(self.priors[class_name])
            MLP = np.where(prob_given_class == max(prob_given_class))[0][0]
            results.append(MLP)


        return results
