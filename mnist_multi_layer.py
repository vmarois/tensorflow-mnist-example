# This file is part of /tensorflow-mnist-example
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# load the MNIST dataset
print('Loading the MNIST dataset from tensorflow.examples.tutorials.mnist, may take some time...')
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
print('Dataset succsfully imported.\n')


print('The dataset consists of', mnist.train.images.shape[0],
      'images stored as arrays of', mnist.train.images.shape[1], 'values (original dimensions of 28 x28).')

# visualize first image of the training set
print('Visualizing the first image of the training set : ')
sample_0 = mnist.train.images[0].reshape(28, 28)
plt.imshow(sample_0, cmap='Greys')
plt.show()
print('That looks like a 3!')

# Now to the interesting bits:

# Define some parameters
learning_rate = 0.001
training_epochs = 100  # how many training cycles we'll go through
batch_size = 100  # size of the batches of the training data

n_classes = 10  # number of classes for the output (-> digits from 0 to 9)
n_samples = mnist.train.num_examples  # number of samples (55 000)
n_input = 784  # shape of one input (array of 784 floats)

n_hidden_1 = 256  # number of neurons for the 1st hidden layer. 256 is common because of the 8-bit color storing method
n_hidden_2 = 256  # number of neurons for the 2nd hidden layer
n_hidden_3 = 256  # number of neurons for the 3rd hidden layer

# We'll use 3 hidden layers. The number of hidden layers is a trade off between speed, cost and accuracy.
# After the output layer, to evaluate the errors between the predictions and the labels, we'll use a loss (cost)
# function.
# Here, we'll just check how many classes we have correctly predicted. We apply an optimizer (Adam) to reduce
# the cost/error at each epoch.


def multilayer_perceptron(x, weights, biases):
    """
    3-layer perceptron for the MNIST dataset.

    :param x: Placeholder for the data input
    :param weights: dict of weights
    :param biases: dict of bias values
    :return: the output layer
    """
    # first hidden layer with RELU activation function
    # X * W + B
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # RELU(X * W + B) -> f(x) = max(0, x)
    layer_1 = tf.nn.relu(layer_1)

    # second hidden layer with RELU activation function
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # third hidden layer with RELU activation function
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # output layer
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer


# define the weights for the nodes of each layer : 784 weights for each node in the first layer,
# then 256 for the 2 next layers, then 10 for the output layer
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),  # matrix of normally distributed random values for H1.
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}

# define the biases for each nodes in each layer : 1 bias per node
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# placeholders for the input data & the labels
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

# All done! Now we can set this up and run the model

pred = multilayer_perceptron(x, weights, biases)

# Define costs and optimization functions. We'll use tf built-in functions
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Some explanation on a batch of 1 sample of the training data
t = mnist.train.next_batch(1)
Xsamp, ysamp = t  # Xsamp is the array of 784 float values, ysamp is an array of length 10 of 0's and contains 1 '1'
# for the corresponding class
# We'll use next_batch() with batch_size to grab a certain number of samples at once

# RUN THE SESSION

with tf.Session() as sess:
    # initialize the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # 'training_epochs' training cycles
    for epoch in range(training_epochs):

        # Cost
        avg_cost = 0.0

        total_batch = int(n_samples/batch_size)

        for i in range(total_batch):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c/total_batch

        print("Epoch : {} -> cost : {:.4f}".format(epoch+1, avg_cost))

    print("Model has completed {} Epochs of training".format(training_epochs))

    # Model evaluations
    correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # Tensor of bool
    correct_predictions = tf.cast(correct_predictions, 'float')  # Cast it to a tensor of floats sor that we can get the
    # mean

    accuracy = tf.reduce_mean(correct_predictions)

    # We now evaluate this accuracy on the test dataset
    print('\nTest Dataset accuracy: {:.4f}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
