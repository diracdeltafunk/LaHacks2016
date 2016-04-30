import tensorflow as tf
import training_input as inp
import tarfile

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

x = tf.placeholder(tf.float32, [None, 19, 19, 3])

w1 = weight([7, 7, 3, 48])
b1 = bias([48])

conv1 = tf.nn.relu(conv2d(x, w1) + b1)
#pool1 = maxpool(conv1)

w2 = weight([5, 5, 48, 32])
b2 = bias([32])

conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)
#pool2 = maxpool(conv2)

w3 = weight([5, 5, 32, 32])
b3 = bias([32])

conv3 = tf.nn.relu(conv2d(conv2, w3) + b3)
#pool3 = maxpool(conv3)

w4 = weight([5, 5, 32, 32])
b4 = bias([32])

conv4 = tf.nn.relu(conv2d(conv3, w4) + b4)
#pool4 = maxpool(conv4)

w5 = weight([19 * 19 * 32, 2048])
b5 = bias([2048])

#pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
flat = tf.reshape(conv4, [-1, 19 * 19 * 32])
dense0 = tf.nn.relu(tf.matmul(flat, w5) + b5)

keep_prob = tf.placeholder(tf.float32)
dense = tf.nn.dropout(dense0, keep_prob)

w6 = weight([2048, 19 * 19])
b6 = bias([19 * 19])

res_flat = tf.nn.softmax(tf.matmul(dense, w6) + b6)

res = tf.reshape(res_flat, [-1, 19, 19])

y1 = tf.placeholder(tf.float32, [None, 19, 19])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y1 * tf.log(res), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# accuracy
pos_real_move = tf.argmax(tf.reshape(y1, [-1, 19 * 19]), 1)
flattened_res = tf.reshape(res, [-1, 19 * 19])
percent_predicted = tf.slice(flattened_res, pos_real_move, [-1, 1])
predicted_tiled = tf.tile(tf.reshape(percent_predicted, [-1, 1, 1]), [1, 1, 19 * 19])
correct_prediction = tf.reduce_sum(tf.where(tf.greater_equal(flattened_res, predicted_tiled)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

tar = tarfile.open("pro.tar.gz", 'r:gz')

with open('filenames.txt','r') as filenames:
    for num, line in enumerate(filenames):
        batch = inp.getdata(tar,line[:-1])
        if num % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y1: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (num, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y1: batch[1], keep_prob: 0.5})
