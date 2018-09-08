# Single Layer Perceptron model to find weights and bias for AND and OR function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


n_features = 2
learning_rate = 0.01
# AND

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
y = np.array([0, 0, 1, 0], np.float32).reshape([4, 1])

xph = tf.placeholder(tf.float32, shape=[4, 2])
yph = tf.placeholder(tf.float32, shape=[4, 1])

w = tf.Variable(tf.zeros([n_features, 1], tf.float32))
b = tf.Variable(tf.zeros([1, 1], tf.float32))

y_hat = tf.sigmoid(tf.add(tf.matmul(xph, w), b))
error = yph - y_hat
delta_w = tf.matmul(tf.transpose(xph), error)
delta_b = tf.reduce_sum(error, 0)
new_w = w + learning_rate * delta_w
new_b = b + learning_rate * delta_b

step = tf.group(w.assign(new_w), b.assign(new_b))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for k in range(1000):
    sess.run([step], feed_dict={xph: x, yph: y})

w = np.squeeze(sess.run(w))
b = np.squeeze(sess.run(b))

plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1] + 0.2)])
plot_y = -1/w[1] * (w[0] * plot_x + b).reshape([2, -1])
plot_y = np.squeeze(plot_y)

print("W: ", str(w))
print("b: ", str(b))
print("plot_y: ", str(plot_y))

new_y = np.reshape(y, [1, 4])
plt.scatter(x[:, 0], x[:, 1], c=new_y[0], s=100, cmap='viridis')
plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.25])
plt.show()


# OR

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
y = np.array([0, 1, 1, 1], np.float32).reshape([4, 1])

xph = tf.placeholder(tf.float32, shape=[4, 2])
yph = tf.placeholder(tf.float32, shape=[4, 1])

w = tf.Variable(tf.zeros([n_features, 1], tf.float32))
b = tf.Variable(tf.zeros([1, 1], tf.float32))

y_hat = tf.sigmoid(tf.add(tf.matmul(xph, w), b))
error = yph - y_hat
delta_w = tf.matmul(tf.transpose(xph), error)
delta_b = tf.reduce_sum(error, 0)
new_w = w + learning_rate * delta_w
new_b = b + learning_rate * delta_b

step = tf.group(w.assign(new_w), b.assign(new_b))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for k in range(1000):
    sess.run([step], feed_dict={xph: x, yph: y})

w = np.squeeze(sess.run(w))
b = np.squeeze(sess.run(b))

plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1] + 0.2)])
plot_y = -1/w[1] * (w[0] * plot_x + b).reshape([2, -1])
plot_y = np.squeeze(plot_y)

print("W: ", str(w))
print("b: ", str(b))
print("plot_y: ", str(plot_y))

new_y = np.reshape(y, [1, 4])
plt.scatter(x[:, 0], x[:, 1], c=new_y[0], s=100, cmap='viridis')
plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.25])
plt.show()
