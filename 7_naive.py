import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target
X = data.drop("Species", axis=1)
y = data['Species']
# Encoding the Species column to get numerical class
le = LabelEncoder()
y = le.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#Predicting the test set results
y_pred = classifier.predict(X_test)
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

data['Species'] = iris.target
X = data.drop("Species", axis=1)
y = data['Species']
# Encoding the Species column to get numerical class
le = LabelEncoder()
y = le.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#Predicting the test set results
y_pred = classifier.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy of Prediction on Iris Flower is: {accuracy}")
print("Training set size:", X_train.shape[0], "Test set size:", X_test.shape[0])






















































# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# Sample dataset (Make sure to replace this with actual data)
# Assuming X is a feature matrix and y is a target variable
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Binary classification


# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Fitting Naive Bayes classifier to the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)




