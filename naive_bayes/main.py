from utils import NaiveBayesClassifier
import random
import numpy as np

def run():
    nb_classifier = NaiveBayesClassifier()
    filename = 'smalldigits.csv'

    # get input and break it into lists
    reviews = nb_classifier.parse_input(filename)


    #randomise and split data into training and test data
    train_data, test_data = nb_classifier.split_train_test(reviews, np.round(len(reviews)*0.8))

    train_y , train_x = nb_classifier.split_feature_label(train_data,  delimiter=',')
    test_y , test_x = nb_classifier.split_feature_label(test_data,  delimiter=',')

    nb_classifier.train(train_x, train_y)

    inference = nb_classifier.classify(test_x, test_y, error=True)

    print(test_y)
    print(inference)






if __name__ == '__main__':
    run()
