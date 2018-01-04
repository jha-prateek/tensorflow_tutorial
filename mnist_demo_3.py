import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False, validation_size=0)

l1 = 500
l2 = 500
l3 = 500

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.truncated_normal([784, l1], stddev=0.1))
b1 = tf.Variable(tf.zeros([l1]))
w2 = tf.Variable(tf.truncated_normal([l1, l2], stddev=0.1))
b2 = tf.Variable(tf.zeros([l2]))
w3 = tf.Variable(tf.truncated_normal([l2, l3], stddev=0.1))
b3 = tf.Variable(tf.zeros([l3]))
w4 = tf.Variable(tf.truncated_normal([l3, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

xx = tf.reshape(x, [-1, 784])
y1 = tf.nn.relu(tf.matmul(xx, w1) + b1)
y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
y_logits = tf.matmul(y3, w4) + b4
y_pred = tf.nn.softmax(y_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y)
cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
    print('Epoch: ', i+1)

print(sess.run([accuracy], feed_dict={x:mnist.test.images, y:mnist.test.labels}))