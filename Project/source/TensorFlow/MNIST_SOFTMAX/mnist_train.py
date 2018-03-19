

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import time
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

start_time = time.time()

# mnist = input_data.read_data_sets("MNIST_data1/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data1/", one_hot=True, validation_size=40)

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)

x = tf.placeholder(tf.float32, [None, 784],name='x')

W = tf.Variable(tf.zeros([784, 10]),name='W')
b = tf.Variable(tf.zeros([10]),name='b')
y = tf.nn.softmax(tf.matmul(x, W) + b,name='y')
y_ = tf.placeholder(tf.float32, [None, 10],name='y_')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# save summaries for visualization
tf.summary.histogram('weights', W)
tf.summary.histogram('max_weight', tf.reduce_max(W))
tf.summary.histogram('bias', b)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)
# merge all summaries into one op
merged=tf.summary.merge_all()
trainwriter = tf.summary.FileWriter('data/mnist_mode'+'/logs/train',sess.graph)

# Launching TensorBoard
# To run TensorBoard, use the following command (alternatively python -m tensorboard.main)
# tensorboard --logdir=J:\AllworksTT\5542TensorFlow\ICP8\MNIST_SOFTMAX\data\mnist_mode\logs
# http://DESKTOP-H4M791U:6006

init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    trainwriter.add_summary(summary, i)

# model export path
tf.add_to_collection('variable', W)
tf.add_to_collection('variable', b)
export_path = 'data/mnist_mode'
print('Exporting trained model to', export_path)

#
# tf.reset_default_graph()
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'scores': y})})

model_exporter.export(export_path, tf.constant(1), sess)


##################################
# get saved weights
W = tf.get_collection('variable')[0]
b = tf.get_collection('variable')[1]

# placeholders for test images and labels
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

# predict equation
y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')

# compare predicted label and actual label
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy op
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accu = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(accu)
##################################


print("Execution time = "+str(time.time() - start_time)+ " seconds")
