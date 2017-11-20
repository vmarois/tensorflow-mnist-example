# Tensorflow project - Building a dense neural network for the MNIST dataset #

A Python sample project implementing an example of a dense neural network for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Description

This project has been done following Jose Portilla's [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/) course on Udemy.
One of the section of this course is dedicated to Tensorflow, and provides the example of a dense neural network for the MNIST dataset.

Hence, several `Python` scripts are available in this repo :

* `tf_basics.py` shows some of the Tensorflow basics (TF constants, operations & sessions)
* `mnist_multi_layer.py` implements a 3-layers perceptron, with 256 nodes on each layer. This model is then trained and
we evaluate its accuracy using the test dataset. This is a rather complete example that shows how to use tf.Session()
* `tf_contriblearn` implements the same neural network, but uses the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)
and [tensorflow.contrib.learn](https://www.tensorflow.org/api_guides/python/contrib.learn), which is a high level API for learning with TensorFlow.

## Requirements

* Python 3 (developed with Python 3.6.2)
* Tensorflow (developed with v1.3.0)
* Sklearn (0.19) for model performances assessment
* Matplotlib & Seaborn for visualization
* Numpy

Just run

    pip3 install -r requirements.txt

to install all required packages.


## Model evaluation

We use the [Adam optimization algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) & the [softmax cross entropy](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits) as the cost function.
After 100 training epochs with a learning rate of 0.001, the model achieves an accuracy of 96.8 % (world best is > 99.5 %).