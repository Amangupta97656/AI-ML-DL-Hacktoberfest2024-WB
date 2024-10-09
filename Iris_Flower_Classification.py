# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (species)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier with the training data
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Output the results
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict the species of a new flower sample (user-defined input)
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example data (sepal length, sepal width, petal length, petal width)
prediction = clf.predict(new_sample)

print(f"Predicted Class for the new sample: {iris.target_names[prediction][0]}")
