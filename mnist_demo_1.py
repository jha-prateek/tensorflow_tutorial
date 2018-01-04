import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500

classes = 10
batch_size = 100

# MNIST data size = 28 x 28
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def mnist_net(data):
    hidden_l_1 = {'weights' : tf.Variable(tf.random_normal([784, nodes_hl1])),
                  'biases' : tf.Variable(tf.random_normal([nodes_hl1]))}

    hidden_l_2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
                  'biases': tf.Variable(tf.random_normal([nodes_hl2]))}

    hidden_l_3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
                  'biases': tf.Variable(tf.random_normal([nodes_hl3]))}

    output = {'weights': tf.Variable(tf.random_normal([nodes_hl3, classes])),
                  'biases': tf.Variable(tf.random_normal([classes]))}

    # feedforward
    l1 = tf.add(tf.matmul(data, hidden_l_1['weights']), hidden_l_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l_2['weights']), hidden_l_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l_3['weights']), hidden_l_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output['weights']), output['biases'])

    return output

def train(x):
    prediction = mnist_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= prediction, labels= y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(10):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of 50 loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x)