import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, [None, 784])

x_image = tf.reshape(x, [-1, 28, 28, 1])

w1 = weight([7, 7, 3, 48])
b1 = bias([48])

conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
#pool1 = maxpool(conv1)

w2 = weight([5, 5, 48, 32])
b2 = bias([32])

conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)
#pool2 = maxpool(conv2)

w3 = weight([5, 5, 48, 32])
b3 = bias([32])

conv3 = tf.nn.relu(conv2d(conv2, w3) + b3)
#pool3 = maxpool(conv3)

w4 = weight([5, 5, 48, 32])
b4 = bias([32])

conv4 = tf.nn.relu(conv2d(conv3, w4) + b4)
#pool4 = maxpool(conv4)

w5 = weight([19 * 19 * 32, 1024])
b5 = bias([1024])

#pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
flat = tf.reshape(conv4, [-1, 19 * 19 * 32])
dense0 = tf.nn.relu(tf.matmul(flat, w5) + b5)

keep_prob = tf.placeholder(tf.float32)
dense = tf.nn.dropout(dense0, keep_prob)

w4 = weight([1024, 19, 19])
b4 = bias([19, 19])

res = tf.nn.softmax(tf.matmul(dense, w4) + b4)

y1 = tf.placeholder(tf.float32, [None, 19, 19])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y1 * tf.log(res), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).tf.train.A.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(y1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y1: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y1: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y1: mnist.test.labels, keep_prob: 1.0}))




