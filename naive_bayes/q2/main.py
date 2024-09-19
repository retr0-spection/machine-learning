import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nb import NaiveBayesClassifier

df = pd.read_csv('smalldigits.csv')


x_labels = df.iloc[:, :-1]
y_labels = df.iloc[:,-1:]

x_labels.columns = [str(i) for i in range(x_labels.shape[1])]
y_labels.columns = [0]



X_train, X_test, y_train, y_test = train_test_split(x_labels, y_labels, test_size=.2)


naive = NaiveBayesClassifier()


#train data
model = naive.train_model(X_train, y_train)
predictions = naive.classify(X_test, y_test)

print('----------------------------------------------------')

true_labels = [row.to_list()[0] for index, row in y_test.iterrows()]
conf_matrix = confusion_matrix(true_labels, predictions)

accuracy = np.trace(conf_matrix) /np.sum(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
print('accuracy:', accuracy)
plt.show()
