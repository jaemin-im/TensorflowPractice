import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# set strong model weights
W = tf.Variable(5.)

# linear model
hypothesis = X * W

# manual gradient
gradient = tf.reduce_mean(tf.)