import tensorflow as tf
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
from struct import unpack
import seaborn as sns
import matplotlib.pyplot as plt
import time

start_time = time.time()
tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.Session()

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# **** cmd command for tensorboard **** tensorboard --logdir=train
# restore the saved model
new_saver = tf.train.import_meta_graph('data/mnist_model/00000001/export.meta')
new_saver.restore(sess, 'data/mnist_model/00000001/export')

# print to see the restored variables
for v in tf.get_collection('variables'):
    print(v.name)
print(sess.run(tf.global_variables()))

# get saved weights
W = tf.get_collection('variables')[0]
b = tf.get_collection('variables')[1]

# placeholders for test images and labels
x = tf.placeholder(tf.float32, [None, 784],name='x')
y_ = tf.placeholder(tf.float32, [None, 10],name='y_')

# predict equation
y = tf.nn.softmax(tf.matmul(x, W) + b,name='y')

# compare predicted label and actual label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# accuracy op
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accu=sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("\nSoftmax regression accuracy: " + str(accu))
print("Softmax execution time = "+str(time.time() - start_time)+ " seconds")



# ------------------------------ for logistic regression of mnist data ---------------
# code from https://gist.github.com/mGalarnyk/aa79813d7ecb0049c7b926d53f588ae1

def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

# load training images and labels, load test images and labels
train_img, train_lbl = loadmnist('mnist_unzip/train-images-idx3-ubyte'
                                 , 'mnist_unzip/train-labels-idx1-ubyte')
test_img, test_lbl = loadmnist('mnist_unzip/t10k-images-idx3-ubyte'
                               , 'mnist_unzip/t10k-labels-idx1-ubyte')


print("\nTraining Image shape: "+str(train_img.shape))
print("Training label shape: "+str(train_lbl.shape))
print("Test label shape: "+str(test_img.shape))
print("Test label shape: "+str(test_lbl.shape))

# start execution time after loading the data
start_time1 = time.time()

#init logistic regression model
logReg = LogisticRegression(solver='lbfgs')
linReg = LinearRegression()
linReg.fit(train_img,train_lbl)
logReg.fit(train_img, train_lbl)
logReg.predict(test_img)

# get logistic regression acurracy, correct predictions / total number of data points
acc = logReg.score(test_img,test_lbl)
print("\nlogistic regression accuracy: "+str(acc))
print("Logistic execution time = "+str(time.time() - start_time1)+ " seconds")

# make confusion matrix with seaborn and display it
predictions = logReg.predict(test_img)
cm = metrics.confusion_matrix(test_lbl, predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(9,9))
sns.heatmap(cm_normalized, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {:.3f}'.format(acc)
plt.title(all_sample_title, size = 15);

plt.show()

