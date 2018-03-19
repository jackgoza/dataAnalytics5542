import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# matplotlib inline

# load_ext autoreload
# autoreload 2

start_time = time.time()

MNIST = input_data.read_data_sets("MNIST_data1/", one_hot=True, validation_size=40)
# Define parameters for linear model
learning_rate = 0.01
batch_size = 28
n_epochs = 500

# Create placeholders
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

# Create weights and bias
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1,10]), name='bias')

# calculate scores
logits1 = tf.matmul(X, w) + b

# Entropy cost function and loss
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=Y)
loss = tf.reduce_mean(entropy)

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Run optimization and test
loss_history = []
acc_history = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    n_batches = int(MNIST.train.num_examples / batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            _, lossvalue = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            loss_history.append(lossvalue)

        # Check validation accuracy
        n_v_batches = int(MNIST.validation.num_examples / batch_size)
        total_correct_preds = 0
        for j in range(n_v_batches):
            X_batch, Y_batch = MNIST.validation.next_batch(batch_size)
            _, loss_batch, logits_batch = sess.run([optimizer, loss, logits1], feed_dict={X: X_batch, Y: Y_batch})
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)
        validation_accuracy = total_correct_preds / MNIST.validation.num_examples
        acc_history.append(validation_accuracy)

    # Test the model
    n_batches = int(MNIST.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        logits_batch = sess.run(logits1, feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds = sess.run(accuracy)

    print(" Test accuracy is ", acc_history)
    # "Test accuracy is {0}".format(total_correct_preds / MNIST.test.num_examples)
print("Execution time = "+str(time.time() - start_time)+ " seconds")
# print("Test accuracy is {0}",accuracy)

plt.subplot(2,1,1)
plt.plot(loss_history, '-o', label='Cost value')
plt.title('Training Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(2,1,2)
plt.plot(acc_history, '-o', label='Accuracy value')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(10, 10)
plt.show()

# print("Softmax execution time = "+str(time.time() - start_time)+ " seconds")