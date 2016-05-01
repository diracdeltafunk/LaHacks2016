import tensorflow as tf
import training_input_cole as inp
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

x = tf.placeholder(tf.float32, [None, 19, 19, 3])
batch_size = 50
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
flat = tf.reshape(conv4, [batch_size, 19 * 19 * 32])
dense0 = tf.nn.relu(tf.matmul(flat, w5) + b5)

keep_prob = tf.placeholder(tf.float32)
dense = tf.nn.dropout(dense0, keep_prob)

w6 = weight([2048, 19 * 19])
b6 = bias([19 * 19])

res_flat = tf.nn.softmax(tf.matmul(dense, w6) + b6)

res = tf.reshape(res_flat, [batch_size, 19, 19])

y1 = tf.placeholder(tf.float32, [None, 19, 19])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y1 * tf.log(res), reduction_indices=[1, 2]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy
y1_flat = tf.reshape(y1, [batch_size, 19 * 19])
pos_real_move = tf.argmax(y1_flat, 1)
percent_predicted = tf.gather(tf.reshape(res_flat, [19 * 19 * batch_size]), tf.add((19 * 19) * tf.to_int64(tf.range(0, batch_size, 1)), pos_real_move))
#percent_predicted = tf.diag_part(tf.gather(tf.transpose(res_flat), pos_real_move))
predicted_tiled = tf.tile(tf.reshape(percent_predicted, [batch_size, 1]), [1, 19 * 19])
correct_prediction = tf.reduce_sum(tf.to_int64(tf.greater_equal(res_flat, predicted_tiled)), reduction_indices=[1])
#correct_prediction = tf.equal(tf.argmax(res_flat, 1), tf.argmax(y1_flat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

tar = tarfile.open("pro.tar.gz", 'r:gz')
saver = tf.train.Saver()
saver.restore(sess, 'saved_network.ckpt')
with open('filenames.txt','r') as filenames:
    for num, line in enumerate(filenames):
#        print(line)
        if num < 10500:
            continue
        bad, batch_in, batch_out = inp.getdata(tar, line[:-1])
#        print(batch_out.shape)
#        print(batch_out[20])
        if not bad:
            if num % 100 == 0:
#            print(res_flat.eval(feed_dict={x: batch_in, y1: batch_out, keep_prob: 1.0}))
                train_accuracy = accuracy.eval(feed_dict={x: batch_in, y1: batch_out, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (num, train_accuracy))
            train_step.run(feed_dict={x: batch_in, y1: batch_out, keep_prob: 0.5})
            if num % 5000 == 4999:
                save_path = saver.save(sess, 'saved_network.ckpt')
    save_path = saver.save(sess, 'saved_network1.ckpt')