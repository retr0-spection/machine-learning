import pandas as pd
import numpy as np
from utils import NaiveBayesGaussian
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('banknote_authentication.csv', sep=';')

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=.2)

# fit all the features to gaussian distribution
#

nb = NaiveBayesGaussian()

nb.fit(X_train, y_train)

results = nb.classify(X_test)
print('----------------------------------------------------')

true_labels = y_test['class'].to_list()
conf_matrix = confusion_matrix(true_labels, results)
size = X_test.shape[0]

accuracy = np.trace(conf_matrix) /np.sum(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
print('accuracy:', accuracy)
plt.show()
