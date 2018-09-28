import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Multi Layer Perceptron for XOR

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
Y = np.array([[0], [1], [1], [0]], np.float32)

# Neural Network Parameters

N_STEPS = 100000
N_EPOCH = 5000
N_TRAINING = len(X)

N_INPUT_NODES = 2
N_HIDDEN_NODES = 2
N_OUTPUT_NODES = 1
LEARNING_RATE = 0.05

x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="y-input")

theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1), name="theta1")
theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES, N_OUTPUT_NODES], -1, 1), name="theta2")

bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")

layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)

cost = tf.reduce_mean(tf.square(Y - output))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(N_STEPS):
    sess.run(train_step, feed_dict={x_: X, y_: Y})
    if i % N_EPOCH == 0:
        print('Batch ', i)
        print('Inference ', sess.run(output, feed_dict={x_: X, y_: Y}))
        print('Cost ', sess.run(cost, feed_dict={x_: X, y_: Y}))

theta1 = np.array(sess.run(theta1), np.float32)
theta2 = np.array(sess.run(theta2), np.float32)
bias1 = np.array(sess.run(bias1), np.float32)
bias2 = np.array(sess.run(bias2), np.float32)

print("Theta 1", theta1)
print("Theta 2", theta2)
print("Bias 1", bias1)
print("Bias 2", bias2)


plot_x = np.array([np.min(X[:, 0] - 0.2), np.max(X[:, 1] + 0.2)])
plot_y1 = -1/theta1[1][0] * (theta1[0][0] * plot_x + (bias2[0]/theta2[0]) + bias1[0]).reshape([2, -1])
plot_y1 = np.squeeze(plot_y1)
plot_y2 = -1/theta1[1][1] * (theta1[0][1] * plot_x + (bias2[0]/theta2[1]) + bias1[1]).reshape([2, -1])
plot_y2 = np.squeeze(plot_y2)

#print("plot_y: ", str(plot_y))

new_y = np.reshape(Y, [1, 4])
plt.scatter(X[:, 0], X[:, 1], c=new_y[0], s=100, cmap='viridis')
plt.plot(plot_x, plot_y1, color='k', linewidth=2)
plt.plot(plot_x, plot_y2, color='k', linewidth=2)
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.25])
plt.show()
