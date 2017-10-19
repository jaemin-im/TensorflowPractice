import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# set strong model weights
W = tf.Variable(5.)

# linear model
hypothesis = X * W

# manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# get gradients
gvs = optimizer.compute_gradients(cost)

# apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# launch the graph in a session
sess = tf.Session
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
