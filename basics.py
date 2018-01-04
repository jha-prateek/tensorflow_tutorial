import tensorflow as tf

sess = tf.Session()

hello = tf.constant('Hello AI')
print(sess.run(hello))

a = tf.constant(5)
b = tf.constant(7)
print(sess.run(a+b))

x = tf.placeholder(tf.int16)
y = tf.placeholder(tf.int16)
div = tf.div(x,y)
print(sess.run(div, feed_dict={x:12, y:5}))

m = tf.constant([[2,3],[3,2]])
n = tf.constant([[3,2],[2,3]])
add = tf.add(m,n)
print(sess.run(add))