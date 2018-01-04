import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

Y_logits = tf.matmul(tf.reshape(X, [-1, 784]), w) + b

# Activation function
Y = tf.nn.softmax(Y_logits)

# Cross Entropy formula
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))

# Accuarcy % (dont know how???)
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize all weigts and biases
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(100):
    # division of batches
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = {X: batch_x, Y_: batch_y}

    # training process
    sess.run(train_step, feed_dict=train_data)
    print('Epoch: ', i+1)
# a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

# Checking accuracy and loss on Test data
test_data = {X:mnist.test.images, Y_:mnist.test.labels}
a_, c_ = sess.run([accuracy, cross_entropy], feed_dict=test_data)
print(a_,c_)
sess.close()