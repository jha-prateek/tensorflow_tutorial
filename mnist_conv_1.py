import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False, validation_size=0)

# depth of weights layer wise
l1 = 4
l2 = 8
l3 = 12
l4 = 200

keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# desining filters/weights for differnet layers
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, l1], stddev=0.1)) # 5x5 patch, 1 input channel, K output channels
b1 = tf.Variable(tf.ones([l1])/10) # division by 10 to get floating values
w2 = tf.Variable(tf.truncated_normal([5, 5, l1, l2], stddev=0.1))
b2 = tf.Variable(tf.ones([l2])/10)
w3 = tf.Variable(tf.truncated_normal([4, 4, l2, l3], stddev=0.1))
b3 = tf.Variable(tf.ones([l3])/10)

w4 = tf.Variable(tf.truncated_normal([7 * 7 * 12, l4], stddev=0.1)) # fully connected layer
b4 = tf.Variable(tf.ones([l4])/10)
w5 = tf.Variable(tf.truncated_normal([l4, 10], stddev=0.1)) # final output layer
b5 = tf.Variable(tf.ones([10])/10)

y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)

# reshape the output for Fully Connected layer
y3_reshape = tf.reshape(y3, [-1, 7 * 7 * 12])
y4 = tf.nn.relu(tf.matmul(y3_reshape, w4) + b4)
y4_drop = tf.nn.dropout(y4, keep_prob=keep_prob)
y_logits = tf.matmul(y4, w5) + b5
y_pred = tf.nn.softmax(y_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y)
cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    x_batch, y_batch = mnist.train.next_batch(100)
    feed_data = {x: x_batch, y: y_batch, keep_prob: 0.75}
    sess.run(train_step, feed_dict=feed_data)
    print('Epoch: ', i+1)

print(sess.run([accuracy], feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0}))
sess.close()