# This file is part of /tensorflow-mnist-example
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow.contrib.learn as skflow
import tensorflow as tf

import numpy as np

# IGNORE WARNINGS
tf.logging.set_verbosity(tf.logging.ERROR)
# ----------------

# Load dataset, and separate it into the features and the target.
iris = load_iris()
X = iris['data']
y = iris['target']

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Split the dataset into a training and a test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Build 3 layer DNN with 10, 20, 10 nodes respectively.
classifier = skflow.DNNClassifier(hidden_units=[10, 20, 10],
                                  feature_columns=feature_columns,
                                  n_classes=3,
                                  model_dir="/tmp/iris_model")

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_train)},
    y=np.array(y_train),
    num_epochs=None,
    shuffle=True)

# Train model.
classifier.fit(input_fn=train_input_fn, steps=2000)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_test)},
    y=np.array(y_test),
    num_epochs=1,
    shuffle=False)

# Get the predictions on the test data.
iris_predictions = classifier.predict(input_fn=test_input_fn, as_iterable=False)

# Get the false positives & negatives
print(' Confusion matrix : \n')
print(confusion_matrix(y_test, iris_predictions))
print('\n Classification report : \n')
print(classification_report(y_test, iris_predictions))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
        [5.8, 3.1, 5.0, 1.7]],
    dtype=np.float32)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [iris['target_names'][p] for p in predictions]

print("New Samples, Class Predictions:    {}\n".format(predicted_classes))
