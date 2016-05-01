import tensorflow as tf
import training_input_value as inp
import tarfile

def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x_v = tf.placeholder(tf.float32, [None, 19, 19, 3])
batch_size = 50
w1_v = weight([7, 7, 3, 48])
b1_v = bias([48])

conv1_v = tf.nn.relu(conv2d(x_v, w1_v) + b1_v)
#pool1 = maxpool(conv1)

w3_v = weight([5, 5, 48, 32])
b3_v = bias([32])

conv3_v = tf.nn.relu(conv2d(conv1_v, w3_v) + b3_v)
#pool3 = maxpool(conv3)

w4_v = weight([5, 5, 32, 32])
b4_v = bias([32])

conv4_v = tf.nn.relu(conv2d(conv3_v, w4_v) + b4_v)
#pool4 = maxpool(conv4)

w5_v = weight([19 * 19 * 32, 2048])
b5_v = bias([2048])

#pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
flat_v = tf.reshape(conv4_v, [batch_size_v, 19 * 19 * 32])
dense0_v = tf.nn.relu(tf.matmul(flat, w5_v) + b5_v)

keep_prob_v = tf.placeholder(tf.float32)
dense_v = tf.nn.dropout(dense0_v, keep_prob_v)

w6_v = weight([2048, 2])
b6_v = bias([2])

res_flat_v = tf.nn.softmax(tf.matmul(dense, w6_v) + b6_v)

res_v = tf.reshape(res_flat_v, [batch_size, 2])

y1_v = tf.placeholder(tf.float32, [None, 2])

cross_entropy_v = tf.reduce_mean(-tf.reduce_sum(y1_v * tf.log(res_v), reduction_indices=[1]))
train_step_v = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_v)
correct_prediction_v = tf.equal(tf.argmax(res_v, 1), tf.argmax(y1_v, 1))
accuracy_v = tf.reduce_mean(tf.cast(correct_prediction_v, tf.float32))

sess.run(tf.initialize_all_variables())

tar = tarfile.open("pro.tar.gz", 'r:gz')
saver = tf.train.Saver()
with open('filenames.txt', 'r') as filenames:
    for num, line in enumerate(filenames):
#        print(line)
        if num < 300:
            continue
        bad, batch_in, batch_out = inp.getdata(tar, line[:-1])
#        print(batch_out.shape)
#        print(batch_out[20])
        if not bad:
            if num % 100 == 0:
#            print(res_flat.eval(feed_dict={x: batch_in, y1: batch_out, keep_prob: 1.0}))
                train_accuracy = accuracy.eval(feed_dict={x: batch_in, y1: batch_out, keep_prob: 1.0})
                print("step %d, training accuracy %.4f" % (num, train_accuracy))
            if num % 5000 == 4999:
                saver.save(sess, 'saved_value_network.ckpt')
            train_step.run(feed_dict={x: batch_in, y1: batch_out, keep_prob: 0.5})
    save_path = saver.save(sess, 'saved_value_network_final.ckpt')
